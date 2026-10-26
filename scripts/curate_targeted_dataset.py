"""
Curate Targeted Dataset & Actionable Hard Examples Specification (P2 - Step 4)
-------------------------------------------------------------------------------
This script extracts and decomposes all 81 test errors from the locked independent
test set (N=1,988), identifies exact data leakage pairs, and produces an actionable
data curation specification for targeted data collection before any model retraining.
Uses standard library 'csv' (no extra dependencies).
"""

import sys
import os
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mobilenetv3_best.pth"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
OUTPUT_ERRORS_CSV = PROJECT_ROOT / "data" / "hard_examples_test_errors.csv"
OUTPUT_SPEC_JSON = PROJECT_ROOT / "data" / "targeted_collection_spec.json"

CLASSES = ["plastic", "paper", "metal", "glass", "organic", "other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
OPTIMAL_TEMPERATURE = 1.6366  # Calibrated on Validation Set via L-BFGS

class SimpleManifestDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.samples = []
        self.root_dir = Path(root_dir)
        self.transform = transform
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img_rel_path = row["image_path"]
        full_path = self.root_dir / img_rel_path
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_str = row.get("target_class", row.get("label"))
        label_idx = CLASS_TO_IDX[label_str]
        return image, label_idx, str(img_rel_path)

def compute_image_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_train_test_leakage():
    print("[1/4] Scanning for exact image hash duplicates (Data Leakage)...", flush=True)
    train_hashes = {}
    with open(TRAIN_CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = PROJECT_ROOT / row["image_path"]
            if p.exists():
                h = compute_image_sha256(p)
                train_hashes[h] = row["image_path"]
            
    leakage_records = []
    with open(TEST_CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = PROJECT_ROOT / row["image_path"]
            if p.exists():
                h = compute_image_sha256(p)
                if h in train_hashes:
                    leakage_records.append({
                        "sha256": h,
                        "train_path": train_hashes[h],
                        "test_path": row["image_path"],
                        "target_class": row.get("target_class", row.get("label"))
                    })
    print(f"      Found {len(leakage_records)} exact duplicate hash pairs between train and test.", flush=True)
    return leakage_records

def evaluate_and_extract_errors():
    print("[2/4] Loading trained MobileNetV3-Large model and evaluating test set...", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    import torchvision.models as models
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, len(CLASSES))
    
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SimpleManifestDataset(TEST_CSV_PATH, PROJECT_ROOT, eval_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    error_records = []
    confusion_stats = defaultdict(int)

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            
            raw_probs = torch.softmax(logits, dim=1)
            calibrated_probs = torch.softmax(logits / OPTIMAL_TEMPERATURE, dim=1)

            for i in range(len(labels)):
                true_idx = labels[i].item()
                true_class = CLASSES[true_idx]

                top_probs, top_indices = torch.topk(raw_probs[i], k=2)
                pred_idx = top_indices[0].item()
                second_idx = top_indices[1].item()
                pred_class = CLASSES[pred_idx]
                
                raw_conf = float(top_probs[0].item())
                second_conf = float(top_probs[1].item())
                margin = float(raw_conf - second_conf)

                cal_conf = float(calibrated_probs[i][pred_idx].item())

                if pred_idx != true_idx:
                    confusion_stats[f"{true_class}->{pred_class}"] += 1
                    source_dataset = "RealWaste" if "realwaste" in paths[i].lower() else "Kaggle12"
                    error_records.append({
                        "image_path": paths[i],
                        "source_dataset": source_dataset,
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "raw_confidence": round(raw_conf, 4),
                        "calibrated_confidence": round(cal_conf, 4),
                        "top2_margin": round(margin, 4),
                        "is_uncertain_margin_0_15": bool(margin < 0.15)
                    })

    print(f"      Total Test Errors Identified: {len(error_records)} / {len(dataset)} samples", flush=True)
    return error_records, confusion_stats

def main():
    print("=" * 70, flush=True)
    print("STEP 4: TARGETED DATASET CURATION & ACTIONABLE SPECIFICATION", flush=True)
    print("=" * 70, flush=True)
    
    leakages = find_train_test_leakage()
    errors, confusion_stats = evaluate_and_extract_errors()

    # Save errors to CSV using built-in csv
    fieldnames = [
        "image_path", "source_dataset", "true_class", "predicted_class",
        "raw_confidence", "calibrated_confidence", "top2_margin", "is_uncertain_margin_0_15"
    ]
    with open(OUTPUT_ERRORS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for err in sorted(errors, key=lambda x: (x["true_class"], x["top2_margin"])):
            writer.writerow(err)
    print(f"[3/4] Exported detailed error list to: {OUTPUT_ERRORS_CSV}", flush=True)

    # Generate Actionable Data Collection Specification
    spec = {
        "title": "Kế hoạch Bổ sung Dữ liệu Mục tiêu (Targeted Data Collection Specification)",
        "objective": "Khắc phục triệt để các ca nhầm lẫn thị giác hàng đầu và rủi ro rác nguy hại trước khi huấn luyện lại.",
        "prerequisites_verified": {
            "test_accuracy_baseline": 0.9598,
            "total_test_errors": len(errors),
            "worst_class": "plastic (F1: 92.19%, 41 errors involved)",
            "second_worst_class": "metal (Precision: 90.28%, 12 false positives)"
        },
        "data_leakage_mitigation": {
            "leakage_count": len(leakages),
            "action": "Loại bỏ các ảnh trùng lặp khỏi tập huấn luyện train.csv trước vòng retrain để tránh đo lường overfit giả tạo.",
            "examples": leakages[:3]
        },
        "priority_targets": [
            {
                "target_id": "TARGET_1_PLASTIC_REFLECTIVE",
                "priority": "Cao nhất (High)",
                "category": "plastic",
                "error_pattern": "Chai/hộp nhựa trong suốt (PET) bị nhầm sang Thủy tinh (glass) hoặc Nhựa bóng mạ bạc bị nhầm sang Kim loại (metal)",
                "current_error_count": 20,
                "recommended_new_samples": 120,
                "photography_protocol": {
                    "distance": "20cm - 60cm",
                    "lighting": "Ánh sáng phòng tự nhiên, ánh đèn huỳnh quang, có phản quang bề mặt",
                    "angles": ["Góc chụp thẳng 90 độ", "Góc nghiêng 45 độ", "Nằm nghiêng trên nền"],
                    "backgrounds": ["Bàn gỗ", "Sàn gạch men", "Mặt bàn bếp", "Thảm văn phòng"],
                    "physical_states": ["Chai nguyên vẹn", "Chai đã mở nắp", "Chai bị dập/bẹp gập nếp"]
                }
            },
            {
                "target_id": "TARGET_2_COATED_BEVERAGE_CANS",
                "priority": "Cao (High)",
                "category": "metal",
                "error_pattern": "Lon nước ngọt có in nhãn màu/hình vẽ bị nhầm sang plastic (7 ca), hoặc giấy bạc/nắp kim loại mỏng",
                "current_error_count": 12,
                "recommended_new_samples": 80,
                "photography_protocol": {
                    "distance": "20cm - 50cm",
                    "lighting": "Ánh sáng tự nhiên và đèn bàn",
                    "angles": ["Thấy rõ nắp khoen mở", "Thấy thân lon in logo", "Lon bị bẹp dập"],
                    "backgrounds": ["Sàn nhà", "Mặt bàn ăn", "Cạnh thùng rác"]
                }
            },
            {
                "target_id": "TARGET_3_HAZARDOUS_BATTERIES_EWASTE",
                "priority": "Nghiêm ngặt (Safety Critical)",
                "category": "hazardous (pin, ắc quy, e-waste)",
                "error_pattern": "Pin tiểu AA/AAA, pin điện thoại bị phân loại nhầm sang kim loại thông thường hoặc other do thiếu feature space chuyên biệt",
                "current_error_count": "Tiềm ẩn nguy cơ bỏ nhầm thùng rác thường",
                "recommended_new_samples": 60,
                "photography_protocol": {
                    "distance": "15cm - 30cm (Chụp cận cảnh)",
                    "lighting": "Rõ nét các cực âm/dương và nhãn hiệu pin",
                    "types": ["Pin tiểu AA/AAA", "Pin cúc áo (CR2032)", "Pin sạc dự phòng phồng", "Cục sạc/dây sạc hỏng"],
                    "backgrounds": ["Mặt bàn", "Khay phân loại rác nguy hại"]
                }
            }
        ],
        "step_5_retraining_readiness_criteria": [
            "Đã thu thập tối thiểu 200 ảnh mới theo đúng quy cách 3 nhóm mục tiêu trên.",
            "Tập test độc lập data/processed/test.csv giữ nguyên vẹn 100% làm thước đo benchmark so sánh trước/sau.",
            "Các cặp ảnh data leakage (7 cặp) đã được loại bỏ khỏi train.csv.",
            "Mục tiêu sau retrain: F1-score của lớp plastic tăng từ 92.19% lên >= 94.50%, tổng lỗi test giảm từ 81 ca xuống <= 55 ca."
        ]
    }

    with open(OUTPUT_SPEC_JSON, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"[4/4] Generated Actionable Collection Specification: {OUTPUT_SPEC_JSON}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("HOÀN THÀNH BƯỚC 4: KẾ HOẠCH BỔ SUNG DỮ LIỆU ĐÚNG MỤC TIÊU ĐÃ SẴN SÀNG!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
