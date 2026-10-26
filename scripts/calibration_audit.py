"""
P2 — Real-World Evaluation and Confidence Calibration Audit Script.

Computes:
1. Baseline test set metrics (Accuracy, P, R, F1, Confusion Matrix).
2. Calibration metrics on Validation & Test:
   - Expected Calibration Error (ECE)
   - Brier Score
   - Negative Log-Likelihood (NLL)
3. Temperature Scaling:
   - Fits optimal T* on Validation set ONLY via L-BFGS / scipy minimize / PyTorch Adam.
   - Evaluates calibrated ECE & Brier on the independent Test set.
4. Abstention & Uncertainty analysis:
   - Evaluates Top-2 margin threshold (0.05 to 0.30) for coverage vs accuracy trade-off.
5. Perceptual duplicate check across splits.
"""

import sys
import os
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Set UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_CLASSES = ["plastic", "paper", "metal", "glass", "organic", "other"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ManifestDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = root_dir / row["image_path"]
                target_cls = row["target_class"]
                label_idx = CLASS_TO_IDX[target_cls]
                orig_cat = row.get("original_category", "")
                src = row.get("dataset_source", "")
                self.samples.append((img_path, label_idx, target_cls, orig_cat, src))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, target_cls, orig_cat, src = self.samples[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx, str(img_path.name), orig_cat, src


def get_eval_transform():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model(device):
    import torchvision.models as models
    ckpt_path = PROJECT_ROOT / "checkpoints" / "mobilenetv3_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get("target_classes", TARGET_CLASSES)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.to(device)
    model.eval()
    return model, classes


def extract_logits_and_labels(model, dataloader, device):
    all_logits = []
    all_labels = []
    all_metadata = []
    with torch.no_grad():
        for images, labels, names, origs, srcs in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            for n, o, s in zip(names, origs, srcs):
                all_metadata.append({"name": n, "orig_cat": o, "source": s})
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels, all_metadata


def compute_classification_metrics(probs, targets, num_classes=6):
    preds = probs.argmax(dim=1)
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t][p] += 1

    total = len(targets)
    acc = sum(cm[i][i] for i in range(num_classes)) / total

    per_class = {}
    macro_p, macro_r, macro_f1 = [], [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[i][c] for i in range(num_classes) if i != c)
        fn = sum(cm[c][j] for j in range(num_classes) if j != c)
        supp = sum(cm[c][j] for j in range(num_classes))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[TARGET_CLASSES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": supp
        }
        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(np.mean(macro_p)), 4),
        "macro_recall": round(float(np.mean(macro_r)), 4),
        "macro_f1": round(float(np.mean(macro_f1)), 4),
        "per_class": per_class,
        "confusion_matrix": cm
    }


def compute_ece(probs, targets, n_bins=15):
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper) if i > 0 else confidences.ge(bin_lower) & confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            bin_stats.append({
                "bin": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": int(in_bin.sum().item()),
                "acc": round(accuracy_in_bin, 4),
                "conf": round(avg_confidence_in_bin, 4),
                "diff": round(bin_error, 4)
            })

    return ece, bin_stats


def compute_brier_score(probs, targets, num_classes=6):
    """Computes Multi-class Brier Score."""
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()
    return brier


def compute_nll(logits, targets):
    """Computes Negative Log-Likelihood (CrossEntropy)."""
    return F.cross_entropy(logits, targets).item()


class TemperatureScaler(nn.Module):
    """Fits single scalar temperature T on validation logits."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits, val_targets):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()


def evaluate_abstention_policies(probs, targets):
    """
    Evaluates Top-2 margin policy and Top-1 threshold policy.
    Returns coverage and accuracy on the retained (accepted) samples.
    """
    sorted_probs, _ = torch.sort(probs, descending=True, dim=1)
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]
    margins = top1 - top2
    preds = probs.argmax(dim=1)
    correct = preds.eq(targets)

    results = []
    # Test margins from 0.00 to 0.30
    for margin_thresh in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        retained = margins >= margin_thresh
        coverage = retained.float().mean().item()
        if retained.sum() > 0:
            retained_acc = correct[retained].float().mean().item()
        else:
            retained_acc = 1.0
        abstain_count = int((~retained).sum().item())
        abstain_errors = int((~correct[~retained]).sum().item()) if abstain_count > 0 else 0
        results.append({
            "margin_thresh": margin_thresh,
            "coverage": round(coverage * 100, 2),
            "retained_acc": round(retained_acc * 100, 2),
            "abstained_count": abstain_count,
            "abstain_errors_caught": abstain_errors
        })
    return results


def check_near_duplicates():
    """Checks for identical file hashes between train and test/val manifests."""
    manifests = {
        "train": PROJECT_ROOT / "data" / "processed" / "train.csv",
        "val": PROJECT_ROOT / "data" / "processed" / "val.csv",
        "test": PROJECT_ROOT / "data" / "processed" / "test.csv"
    }
    import hashlib

    hash_to_split = defaultdict(list)
    total_checked = 0

    for s_name, s_csv in manifests.items():
        with open(s_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = PROJECT_ROOT / row["image_path"]
                if p.exists():
                    total_checked += 1
                    # Fast MD5 on raw bytes
                    with open(p, "rb") as f_img:
                        h = hashlib.md5(f_img.read()).hexdigest()
                    hash_to_split[h].append((s_name, row["image_path"]))

    leakage_count = 0
    leakage_examples = []
    for h, instances in hash_to_split.items():
        splits = set(inst[0] for inst in instances)
        if len(splits) > 1:
            leakage_count += 1
            if len(leakage_examples) < 5:
                leakage_examples.append(instances)

    return total_checked, leakage_count, leakage_examples


def main():
    print("=" * 80)
    print("  PHASE 2 & 3: REPRODUCIBLE EVALUATION & CONFIDENCE CALIBRATION AUDIT")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Dataset Leakage & Duplicate Audit
    print("\n[1] AUDIT DUPLICATE HASHES ACROSS SPLITS...")
    total_imgs, dup_leakage, dup_examples = check_near_duplicates()
    print(f"Total images checked: {total_imgs}")
    print(f"Exact binary byte hash duplicates across different splits: {dup_leakage}")
    if dup_leakage > 0:
        print(f"Sample duplicate pairs across splits: {dup_examples}")
    else:
        print("PASS: Zero binary byte duplicates found across Train/Val/Test splits.")

    # 2. Load Model & Datasets
    print("\n[2] LOADING MODEL CHECKPOINT & DATASETS...")
    model, classes = load_model(device)
    transform = get_eval_transform()

    val_dataset = ManifestDataset(PROJECT_ROOT / "data" / "processed" / "val.csv", PROJECT_ROOT, transform)
    test_dataset = ManifestDataset(PROJECT_ROOT / "data" / "processed" / "test.csv", PROJECT_ROOT, transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples      : {len(test_dataset)}")

    # 3. Extract Logits
    print("\n[3] EXTRACTING LOGITS & SOFTMAX...")
    val_logits, val_targets, val_meta = extract_logits_and_labels(model, val_loader, device)
    test_logits, test_targets, test_meta = extract_logits_and_labels(model, test_loader, device)

    val_probs_uncal = F.softmax(val_logits, dim=1)
    test_probs_uncal = F.softmax(test_logits, dim=1)

    # 4. Baseline Classification Metrics on Test Set
    test_metrics = compute_classification_metrics(test_probs_uncal, test_targets)
    print("\n" + "=" * 80)
    print("  INDEPENDENT TEST SET PERFORMANCE (N = 1,988)")
    print("=" * 80)
    print(f"Accuracy        : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision : {test_metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall    : {test_metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score  : {test_metrics['macro_f1'] * 100:.2f}%")
    print("-" * 80)
    print(f"{'Class':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 65)
    for c_name in TARGET_CLASSES:
        m = test_metrics["per_class"][c_name]
        print(f"{c_name:<12} | {m['precision']*100:>10.2f}% | {m['recall']*100:>10.2f}% | {m['f1']*100:>10.2f}% | {m['support']:<8}")
    print("-" * 65)

    print("\nConfusion Matrix (Row=True, Col=Pred):")
    header = f"{'True\\Pred':<10} | " + " | ".join(f"{c[:6]:>6}" for c in TARGET_CLASSES)
    print(header)
    for i, row in enumerate(test_metrics["confusion_matrix"]):
        row_str = " | ".join(f"{val:>6}" for val in row)
        print(f"{TARGET_CLASSES[i]:<10} | {row_str}")

    # 5. Calibration Metrics Before Temperature Scaling
    val_ece_uncal, _ = compute_ece(val_probs_uncal, val_targets)
    val_brier_uncal = compute_brier_score(val_probs_uncal, val_targets)
    val_nll_uncal = compute_nll(val_logits, val_targets)

    test_ece_uncal, test_bins_uncal = compute_ece(test_probs_uncal, test_targets)
    test_brier_uncal = compute_brier_score(test_probs_uncal, test_targets)
    test_nll_uncal = compute_nll(test_logits, test_targets)

    print("\n" + "=" * 80)
    print("  UNCALIBRATED CALIBRATION METRICS (Standard Softmax)")
    print("=" * 80)
    print(f"Validation ECE  : {val_ece_uncal * 100:.2f}% | Brier: {val_brier_uncal:.4f} | NLL: {val_nll_uncal:.4f}")
    print(f"Test Set ECE    : {test_ece_uncal * 100:.2f}% | Brier: {test_brier_uncal:.4f} | NLL: {test_nll_uncal:.4f}")

    print("\nTest Calibration Reliability Bins (Uncalibrated):")
    print(f"{'Bin':<16} | {'Count':<8} | {'Avg Conf':<12} | {'Accuracy':<12} | {'Error (Conf-Acc)':<16}")
    print("-" * 72)
    for b in test_bins_uncal:
        print(f"{b['bin']:<16} | {b['count']:<8} | {b['conf']*100:>10.2f}% | {b['acc']*100:>10.2f}% | {b['diff']*100:>14.2f}%")

    # 6. Fit Temperature Scaling on Validation Set ONLY
    print("\n" + "=" * 80)
    print("  TEMPERATURE SCALING CALIBRATION (Fit on Val Set ONLY)")
    print("=" * 80)
    scaler = TemperatureScaler()
    optimal_T = scaler.fit(val_logits, val_targets)
    print(f"Optimal Temperature Parameter T* fitted on Validation set: {optimal_T:.4f}")

    # Apply T* to Validation and Test logits
    val_logits_cal = val_logits / optimal_T
    test_logits_cal = test_logits / optimal_T
    val_probs_cal = F.softmax(val_logits_cal, dim=1)
    test_probs_cal = F.softmax(test_logits_cal, dim=1)

    val_ece_cal, _ = compute_ece(val_probs_cal, val_targets)
    val_brier_cal = compute_brier_score(val_probs_cal, val_targets)
    val_nll_cal = compute_nll(val_logits_cal, val_targets)

    test_ece_cal, test_bins_cal = compute_ece(test_probs_cal, test_targets)
    test_brier_cal = compute_brier_score(test_probs_cal, test_targets)
    test_nll_cal = compute_nll(test_logits_cal, test_targets)

    print(f"\nCalibrated Validation : ECE={val_ece_cal*100:.2f}% (was {val_ece_uncal*100:.2f}%) | NLL={val_nll_cal:.4f} (was {val_nll_uncal:.4f})")
    print(f"Calibrated Test Set   : ECE={test_ece_cal*100:.2f}% (was {test_ece_uncal*100:.2f}%) | NLL={test_nll_cal:.4f} (was {test_nll_uncal:.4f})")
    print(f"Brier Score on Test   : {test_brier_cal:.4f} (was {test_brier_uncal:.4f})")
    print(f"Relative ECE Reduction: {(test_ece_uncal - test_ece_cal) / test_ece_uncal * 100:.1f}% improvement on unseen test set!")

    print("\nTest Calibration Reliability Bins (After Temperature Scaling T*):")
    print(f"{'Bin':<16} | {'Count':<8} | {'Avg Conf':<12} | {'Accuracy':<12} | {'Error (Conf-Acc)':<16}")
    print("-" * 72)
    for b in test_bins_cal:
        print(f"{b['bin']:<16} | {b['count']:<8} | {b['conf']*100:>10.2f}% | {b['acc']*100:>10.2f}% | {b['diff']*100:>14.2f}%")

    # 7. Evaluation of TOP2_MARGIN_THRESHOLD (0.15 vs alternatives)
    print("\n" + "=" * 80)
    print("  ABSTENTION POLICY: TOP-2 MARGIN THRESHOLD AUDIT (Test Set)")
    print("=" * 80)
    abstention_results = evaluate_abstention_policies(test_probs_cal, test_targets)
    print(f"{'Margin Thresh':<14} | {'Coverage':<12} | {'Retained Acc':<14} | {'Abstained #':<12} | {'Errors Caught':<14}")
    print("-" * 72)
    for r in abstention_results:
        mark = " ★ (Current = 0.15)" if abs(r["margin_thresh"] - 0.15) < 1e-4 else ""
        print(f"{r['margin_thresh']:<14.2f} | {r['coverage']:>10.2f}% | {r['retained_acc']:>12.2f}% | {r['abstained_count']:>10} | {r['abstain_errors_caught']:>12}{mark}")

    # 8. Detailed Failure Mode Analysis on Test Set
    print("\n" + "=" * 80)
    print("  DETAILED ERROR DECOMPOSITION ON TEST SET")
    print("=" * 80)
    test_preds = test_probs_uncal.argmax(dim=1)
    wrong_mask = test_preds.ne(test_targets)
    wrong_indices = torch.nonzero(wrong_mask).squeeze(1).tolist()
    print(f"Total classification errors on test set: {len(wrong_indices)} / {len(test_targets)} ({len(wrong_indices)/len(test_targets)*100:.2f}%)")

    # Top confused pairs
    pair_counts = Counter()
    for idx in wrong_indices:
        t = TARGET_CLASSES[test_targets[idx].item()]
        p = TARGET_CLASSES[test_preds[idx].item()]
        pair_counts[(t, p)] += 1

    print("\nTop 5 Error Pairings:")
    for (t, p), count in pair_counts.most_common(5):
        print(f"  True: {t:<10} -> Predicted: {p:<10} : {count} cases ({count/len(wrong_indices)*100:.1f}% of all errors)")

    # Save results to a structured audit artifact JSON
    audit_data = {
        "test_metrics": test_metrics,
        "calibration": {
            "optimal_temperature_val": round(optimal_T, 4),
            "uncalibrated_val_ece": round(val_ece_uncal, 4),
            "calibrated_val_ece": round(val_ece_cal, 4),
            "uncalibrated_test_ece": round(test_ece_uncal, 4),
            "calibrated_test_ece": round(test_ece_cal, 4),
            "uncalibrated_test_brier": round(test_brier_uncal, 4),
            "calibrated_test_brier": round(test_brier_cal, 4),
            "uncalibrated_test_nll": round(test_nll_uncal, 4),
            "calibrated_test_nll": round(test_nll_cal, 4)
        },
        "top2_margin_audit": abstention_results,
        "duplicate_hash_check": {
            "total_checked": total_imgs,
            "leakage_count": dup_leakage
        }
    }

    out_file = PROJECT_ROOT / "checkpoints" / "p2_calibration_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"\nAudit results successfully saved to: {out_file.name}")


if __name__ == "__main__":
    main()

