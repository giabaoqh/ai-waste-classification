#!/usr/bin/env python3
"""
Script chuẩn bị và phân chia tập dữ liệu kết hợp: RealWaste + Kaggle Garbage Classification (12 classes)
Dự án: AI-Based Waste Recognition and Recycling Recommendation System.

Quy trình:
1. Tự động tìm và giải nén tệp zip Kaggle Garbage Classification tại data/raw/kaggle_garbage/.
2. Ánh xạ các danh mục gốc của cả 2 tập dữ liệu về đúng 6 nhóm mục tiêu:
   - plastic  <- Plastic (RealWaste), plastic (Kaggle)
   - paper    <- Paper, Cardboard (RealWaste), paper, cardboard (Kaggle)
   - metal    <- Metal (RealWaste), metal (Kaggle)
   - glass    <- Glass (RealWaste), green-glass, brown-glass, white-glass (Kaggle)
   - organic  <- Food Organics, Vegetation (RealWaste), biological (Kaggle)
   - other    <- Miscellaneous Trash, Textile Trash (RealWaste), battery, clothes, shoes, trash (Kaggle)
3. Kiểm tra tính toàn vẹn của từng ảnh (loại bỏ ảnh hỏng / 0 byte).
4. Phân tầng (Stratified Sampling) theo 6 nhóm với tỷ lệ 70% Train / 15% Val / 15% Test (seed=42).
5. Xuất manifest chuẩn (CSV, JSON) và split_summary.json tại data/processed/.
"""

import os
import sys
import csv
import json
import random
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Đường dẫn dữ liệu
REALWASTE_DIR = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste"
KAGGLE_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle_garbage"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw" / "medical_waste"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Thứ tự lớp chuẩn dự án (đồng bộ 100% với train.py, checkpoint và main.py)
TARGET_CLASSES = ["plastic", "paper", "metal", "glass", "organic", "other", "medical"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {str(idx): cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# Bảng ánh xạ RealWaste (9 lớp -> 6 lớp)
REALWASTE_MAPPING = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "metal": "metal",
    "glass": "glass",
    "food organics": "organic",
    "vegetation": "organic",
    "miscellaneous trash": "other",
    "textile trash": "other"
}

# Bảng ánh xạ Kaggle 12-class (12 lớp -> 6 lớp)
KAGGLE_MAPPING = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "metal": "metal",
    "green-glass": "glass",
    "brown-glass": "glass",
    "white-glass": "glass",
    "glass": "glass",
    "biological": "organic",
    "battery": "other",
    "batteries": "other",
    "clothes": "other",
    "shoes": "other",
    "trash": "other"
}

# Bảng ánh xạ Medical Waste (TrashBox Medical)
MEDICAL_MAPPING = {
    "masks": "medical",
    "gloves": "medical",
    "medicines": "medical",
    "syringe": "medical"
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_kaggle_zip_if_needed(kaggle_dir: Path) -> Path:
    """Tự động tìm và giải nén tệp zip Kaggle nếu có."""
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    extracted_marker = kaggle_dir / "extracted"
    
    # Tìm file zip trong thư mục kaggle_dir hoặc data/raw
    zip_candidates = list(kaggle_dir.glob("*.zip"))
    if not zip_candidates:
        zip_candidates = list((kaggle_dir.parent).glob("*.zip"))
    
    if zip_candidates and not extracted_marker.exists():
        zip_path = zip_candidates[0]
        print(f"📦 Phát hiện tệp zip: {zip_path.name}")
        print(f"⏳ Đang giải nén {zip_path.name} vào {extracted_marker} (Vui lòng chờ giây lát)...")
        extracted_marker.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # An toàn: Kiểm tra Zip Slip
            for member in zip_ref.namelist():
                member_path = (extracted_marker / member).resolve()
                if not str(member_path).startswith(str(extracted_marker.resolve())):
                    raise RuntimeError(f"Cảnh báo bảo mật: Tệp zip chứa đường dẫn nguy hiểm: {member}")
            zip_ref.extractall(extracted_marker)
        print("✅ Giải nén tệp zip Kaggle thành công!")
        return extracted_marker

    if extracted_marker.exists():
        return extracted_marker
    
    return kaggle_dir


def collect_images_from_directory(root_dir: Path, label_mapping: dict, dataset_name: str):
    """Quét và thu thập ảnh hợp lệ từ thư mục dữ liệu."""
    samples = []
    skipped_count = 0

    if not root_dir.exists():
        print(f"⚠️ Cảnh báo: Thư mục {dataset_name} không tồn tại tại: {root_dir}")
        return samples

    # Tìm tất cả thư mục con có thể chứa ảnh
    subdirs = [p for p in root_dir.rglob("*") if p.is_dir()]
    if not subdirs:
        subdirs = [root_dir]

    checked_dirs = set()
    for directory in subdirs:
        dir_name_lower = directory.name.lower().strip()
        matched_target = None
        orig_cat = directory.name

        for raw_name, target in label_mapping.items():
            if dir_name_lower == raw_name.lower():
                matched_target = target
                orig_cat = raw_name
                break

        if not matched_target:
            continue

        checked_dirs.add(directory)
        for img_file in directory.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                if img_file.stat().st_size == 0:
                    skipped_count += 1
                    continue

                # Lưu đường dẫn tương đối so với PROJECT_ROOT
                try:
                    rel_path = img_file.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rel_path = str(img_file)

                samples.append({
                    "image_path": rel_path,
                    "target_class": matched_target,
                    "class_idx": CLASS_TO_IDX[matched_target],
                    "original_category": orig_cat,
                    "dataset_source": dataset_name
                })

    print(f"📁 {dataset_name}: Tìm thấy {len(samples)} ảnh hợp lệ ({skipped_count} ảnh rỗng/lỗi bị bỏ qua).")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Gộp và phân chia tập dữ liệu RealWaste + Kaggle Garbage Classification.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Thư mục lưu trữ manifest kết quả")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỷ lệ Train")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ Validation")
    parser.add_argument("--max-per-class", type=int, default=2800, help="Giới hạn tối đa số ảnh mỗi nhóm để chống mất cân bằng (Mặc định: 2800)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("  CHUẨN BỊ VÀ GỘP TẬP DỮ LIỆU: REALWASTE + KAGGLE + TRASHBOX MEDICAL (7 CLASSES)")
    print("=====================================================================")

    # 1. Thu thập RealWaste
    realwaste_samples = collect_images_from_directory(REALWASTE_DIR, REALWASTE_MAPPING, "RealWaste")

    # 2. Xử lý và thu thập Kaggle
    kaggle_extracted = extract_kaggle_zip_if_needed(KAGGLE_DIR)
    kaggle_samples = collect_images_from_directory(kaggle_extracted, KAGGLE_MAPPING, "Kaggle12")

    # 3. Thu thập Medical Waste (TrashBox)
    medical_samples = collect_images_from_directory(MEDICAL_DIR, MEDICAL_MAPPING, "TrashBox-Medical") if MEDICAL_DIR.exists() else []

    all_samples = realwaste_samples + kaggle_samples + medical_samples
    total_images = len(all_samples)

    if total_images == 0:
        print("\n❌ LỖI: Không tìm thấy bất kỳ ảnh nào để phân chia!")
        print(f"👉 Vui lòng đặt file zip tải từ Kaggle vào: {KAGGLE_DIR}")
        return

    print(f"\n📊 TỔNG CỘNG THU THẬP NGUYÊN BẢN: {total_images} ảnh từ cả 2 tập dữ liệu.")
    
    # Thống kê theo lớp
    class_groups = defaultdict(list)
    for s in all_samples:
        class_groups[s["target_class"]].append(s)

    # Cân bằng dữ liệu (tránh nhóm 'other' chiếm tới 50% làm mô hình thiên lệch)
    if args.max_per_class and args.max_per_class > 0:
        for cls in TARGET_CLASSES:
            original_count = len(class_groups[cls])
            if original_count > args.max_per_class:
                print(f"⚖️ Cân bằng nhóm '{cls}': Cắt giảm từ {original_count} xuống {args.max_per_class} ảnh ngẫu nhiên để chống lệch phân phối.")
                random.shuffle(class_groups[cls])
                class_groups[cls] = class_groups[cls][:args.max_per_class]

    total_balanced_images = sum(len(items) for items in class_groups.values())
    print(f"📊 TỔNG CỘNG SAU CÂN BẰNG: {total_balanced_images} ảnh.")
    for cls in TARGET_CLASSES:
        print(f"   • {cls.upper():<10}: {len(class_groups[cls])} ảnh")

    # 3. Phân tầng (Stratified Sampling)
    train_samples = []
    val_samples = []
    test_samples = []

    for cls in TARGET_CLASSES:
        items = class_groups[cls]
        random.shuffle(items)
        n = len(items)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    # Xáo trộn lại từng split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # 4. Lưu manifest
    splits_data = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

    csv_fields = ["image_path", "target_class", "class_idx", "original_category", "dataset_source"]

    for split_name, samples_list in splits_data.items():
        csv_file = output_dir / f"{split_name}.csv"
        json_file = output_dir / f"{split_name}.json"

        # Ghi CSV
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(samples_list)

        # Ghi JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(samples_list, f, ensure_ascii=False, indent=2)

        print(f"💾 Đã lưu: {csv_file.name} ({len(samples_list)} ảnh)")

    # 5. Lưu summary
    summary = {
        "project": "AI-Based Waste Recognition and Recycling Recommendation System",
        "datasets": ["RealWaste", "Kaggle12"],
        "random_seed": args.seed,
        "total_images": total_images,
        "target_classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "splits": {
            s_name: {
                "count": len(s_list),
                "percentage": round(len(s_list) / total_images * 100, 2),
                "distribution": {cls: sum(1 for x in s_list if x["target_class"] == cls) for cls in TARGET_CLASSES}
            } for s_name, s_list in splits_data.items()
        }
    }

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Cập nhật checkpoints/class_mapping.json đồng bộ
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({
            "target_classes": TARGET_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS
        }, f, indent=2)

    print("\n🎉 HOÀN THÀNH CHUẨN BỊ VÀ PHÂN CHIA TẬP DỮ LIỆU!")
    print(f"👉 File tổng kết lưu tại: {output_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
