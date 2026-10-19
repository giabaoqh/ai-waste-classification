#!/usr/bin/env python3
"""
Script huấn luyện mô hình Baseline (Transfer Learning với MobileNetV3-Large)
Dự án: AI-Based Waste Recognition and Recycling Recommendation System.

Quy trình:
1. Đọc manifests đã tạo từ data/processed/ (train.csv, val.csv, test.csv).
2. Thiết lập 6 lớp mục tiêu theo thứ tự cố định: plastic, paper, metal, glass, organic, other.
3. Sử dụng mô hình MobileNetV3-Large với trọng số tiền huấn luyện ImageNet.
4. Thay thế lớp phân loại cuối cùng (classifier) thành 6 output classes.
5. Huấn luyện 10 epochs, đánh giá sau mỗi epoch trên tập Validation, lưu checkpoint tốt nhất.
6. Đánh giá độc lập 1 lần duy nhất trên tập Test (713 ảnh) với confusion matrix và các chỉ số chi tiết.
7. Lưu checkpoint và kết quả đánh giá vào thư mục checkpoints/ (được .gitignore bảo vệ).
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đường dẫn thư mục dự án
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Thứ tự lớp cố định theo đặc tả dự án
TARGET_CLASSES = ["plastic", "paper", "metal", "glass", "organic", "other"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(TARGET_CLASSES)}

# Chuẩn chuẩn hóa ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    """Cố định seed để đảm bảo 100% tính tái lập (reproducibility)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class WasteDataset(Dataset):
    """Custom Dataset nạp ảnh từ manifest CSV."""
    def __init__(self, csv_file: Path, root_dir: Path, class_to_idx: dict, transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not csv_file.exists():
            raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp manifest tại: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row["image_path"]
                target_cls = row["target_class"]
                if target_cls not in self.class_to_idx:
                    raise ValueError(f"Lỗi: Lớp '{target_cls}' không nằm trong TARGET_CLASSES.")
                
                img_path = self.root_dir / img_rel
                if not img_path.exists():
                    raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại: {img_path}")
                
                label_idx = self.class_to_idx[target_cls]
                self.samples.append((img_path, label_idx, target_cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, _ = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Lỗi đọc ảnh tại {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def get_transforms(input_size: int = 224):
    """Tạo transforms cho Train (Augmentation) và Val/Test (Deterministic)."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


def build_model(num_classes: int = 6):
    """Khởi tạo MobileNetV3-Large với ImageNet pretrained weights và thay thế classifier head."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    
    # Thay thế lớp linear cuối cùng (classifier[3]) từ 1000 lớp sang 6 lớp
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Huấn luyện 1 epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Đánh giá loss và accuracy trên validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def compute_metrics(y_true, y_pred, num_classes=6):
    """Tính toán ma trận nhầm lẫn và các chỉ số Precision, Recall, F1 (Macro & Weighted)."""
    # Khởi tạo Confusion Matrix (num_classes x num_classes)
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    total_samples = len(y_true)
    correct_samples = sum(cm[i][i] for i in range(num_classes))
    overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    per_class_metrics = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    class_supports = []

    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[i][c] for i in range(num_classes) if i != c)
        fn = sum(cm[c][j] for j in range(num_classes) if j != c)
        support = sum(cm[c][j] for j in range(num_classes))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        cls_name = IDX_TO_CLASS[c]
        per_class_metrics[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)
        class_supports.append(support)

    macro_precision = sum(macro_precisions) / num_classes
    macro_recall = sum(macro_recalls) / num_classes
    macro_f1 = sum(macro_f1s) / num_classes
    weighted_f1 = sum(f1 * sup for f1, sup in zip(macro_f1s, class_supports)) / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm
    }


def evaluate_test_set(model, test_loader, device):
    """Thực hiện đánh giá độc lập trên Test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())

    return compute_metrics(all_targets, all_preds, num_classes=len(TARGET_CLASSES))


def print_evaluation_report(results):
    """In bảng kết quả đánh giá chi tiết ra console."""
    print("\n" + "=" * 80)
    print("  KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (TEST EVALUATION RESULTS)")
    print("=" * 80)
    print(f"Test Accuracy    : {results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall     : {results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score   : {results['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score: {results['weighted_f1'] * 100:.2f}%")
    print("-" * 80)
    print(f"{'Class':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 65)
    for cls_name in TARGET_CLASSES:
        metrics = results["per_class"][cls_name]
        print(f"{cls_name:<12} | {metrics['precision']*100:>10.2f}% | {metrics['recall']*100:>10.2f}% | {metrics['f1_score']*100:>10.2f}% | {metrics['support']:<8}")
    print("-" * 65)

    print("\nMA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("Dòng = Nhãn thực tế (True), Cột = Nhãn dự đoán (Pred)")
    header = f"{'True \\ Pred':<12} | " + " | ".join(f"{c[:6]:>6}" for c in TARGET_CLASSES)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(results["confusion_matrix"]):
        row_str = " | ".join(f"{val:>6}" for val in row)
        print(f"{TARGET_CLASSES[i]:<12} | {row_str}")
    print("-" * len(header))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Baseline MobileNetV3-Large trên tập dữ liệu RealWaste."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện (Mặc định: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch (Mặc định: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate ban đầu (Mặc định: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (Mặc định: 42)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị tính toán")
    parser.add_argument("--num-workers", type=int, default=0, help="Số workers nạp dữ liệu (Mặc định: 0 an toàn trên Windows)")
    return parser.parse_args()


def run_training():
    args = parse_arguments()
    set_seed(args.seed)

    # Xác định thiết bị tính toán
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Lỗi: Yêu cầu CUDA nhưng CUDA không khả dụng trên máy.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("================================================================================")
    print("  HUẤN LUYỆN MÔ HÌNH BASELINE PHÂN LOẠI RÁC THẢI (PHASE 5.4.2B)")
    print("================================================================================")
    print(f"Kiến trúc mô hình: MobileNetV3-Large (Pretrained ImageNet)")
    print(f"Số lớp phân loại : {len(TARGET_CLASSES)} ({', '.join(TARGET_CLASSES)})")
    print(f"Thiết bị tính toán: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM khả dụng    : {vram:.2f} GB")
    print(f"Tham số          : Epochs={args.epochs}, Batch size={args.batch_size}, LR={args.lr}, Seed={args.seed}")
    print("--------------------------------------------------------------------------------")

    train_csv = PROCESSED_DATA_DIR / "train.csv"
    val_csv = PROCESSED_DATA_DIR / "val.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"

    train_transform, eval_transform = get_transforms(input_size=224)

    train_dataset = WasteDataset(train_csv, PROJECT_ROOT, CLASS_TO_IDX, transform=train_transform)
    val_dataset = WasteDataset(val_csv, PROJECT_ROOT, CLASS_TO_IDX, transform=eval_transform)
    test_dataset = WasteDataset(test_csv, PROJECT_ROOT, CLASS_TO_IDX, transform=eval_transform)

    print(f"Dữ liệu nạp từ manifest:")
    print(f"  - Train     : {len(train_dataset)} mẫu")
    print(f"  - Validation: {len(val_dataset)} mẫu")
    print(f"  - Test      : {len(test_dataset)} mẫu")
    print(f"  - Tổng cộng : {len(train_dataset) + len(val_dataset) + len(test_dataset)} mẫu")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(num_classes=len(TARGET_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINT_DIR / "mobilenetv3_best.pth"

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    training_history = []

    print("\nBẮT ĐẦU HUẤN LUYỆN (10 EPOCHS):")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'LR':<10} | {'Thời gian':<10}")
    print("-" * 85)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and batch_size > 16:
                print(f"\n[CẢNH BÁO OOM] VRAM cạn kiệt ở batch_size={batch_size}. Đang giải phóng bộ nhớ và chuyển sang batch_size=16...")
                torch.cuda.empty_cache()
                batch_size = 16
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                continue
            else:
                raise e

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        is_best = False
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            is_best = True
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            # Lưu checkpoint tốt nhất
            checkpoint_payload = {
                "model_architecture": "mobilenet_v3_large",
                "state_dict": model.state_dict(),
                "target_classes": TARGET_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
                "input_size": [3, 224, 224],
                "normalization": {
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD
                },
                "best_epoch": best_epoch,
                "best_val_metrics": {
                    "val_loss": round(best_val_loss, 4),
                    "val_accuracy": round(best_val_acc, 4)
                },
                "seed": args.seed,
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": batch_size,
                    "learning_rate": args.lr,
                    "optimizer": "Adam",
                    "scheduler": "CosineAnnealingLR",
                    "loss": "CrossEntropyLoss"
                }
            }
            torch.save(checkpoint_payload, best_checkpoint_path)

        star = " ★ (Best)" if is_best else ""
        print(f"{epoch:<8} | {train_loss:<12.4f} | {train_acc*100:<11.2f}% | {val_loss:<12.4f} | {val_acc*100:<11.2f}% | {current_lr:<10.6f} | {elapsed:<8.1f}s{star}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 6),
            "time_seconds": round(elapsed, 2),
            "is_best": is_best
        })

    total_train_time = time.time() - start_train_time
    print("-" * 85)
    print(f"Tổng thời gian huấn luyện: {total_train_time / 60:.2f} phút.")
    print(f"Mô hình tốt nhất được chọn tại Epoch {best_epoch}: Val Acc = {best_val_acc*100:.2f}%, Val Loss = {best_val_loss:.4f}.")
    print(f"Checkpoint đã lưu tại: {best_checkpoint_path}")

    # 1. Lưu training history
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    print(f"Lịch sử huấn luyện đã lưu tại: {history_path.name}")

    # 2. Lưu class mapping
    mapping_path = CHECKPOINT_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_classes": TARGET_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS
        }, f, indent=2, ensure_ascii=False)
    print(f"Bảng ánh xạ lớp đã lưu tại: {mapping_path.name}")

    # 3. Đánh giá độc lập trên Test Set bằng mô hình tốt nhất
    print(f"\n[ĐÁNH GIÁ ĐỘC LẬP] Đang tải checkpoint tốt nhất (Epoch {best_epoch}) để kiểm thử trên Test set...")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])

    test_results = evaluate_test_set(model, test_loader, device)
    test_results["best_epoch"] = best_epoch
    test_results["best_val_acc"] = round(best_val_acc, 4)
    test_results["best_val_loss"] = round(best_val_loss, 4)
    test_results["total_train_time_seconds"] = round(total_train_time, 2)

    # Lưu kết quả test evaluation
    eval_path = CHECKPOINT_DIR / "test_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"Kết quả kiểm thử đã lưu tại: {eval_path.name}")

    # In báo cáo chi tiết
    print_evaluation_report(test_results)


if __name__ == "__main__":
    run_training()
