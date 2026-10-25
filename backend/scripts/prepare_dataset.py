#!/usr/bin/env python3
"""
Script chuẩn bị và phân chia tập dữ liệu RealWaste (Train / Validation / Test)
Dự án: AI-Based Waste Recognition and Recycling Recommendation System.

Quy trình:
1. Ánh xạ 9 danh mục gốc của RealWaste sang đúng 6 nhóm mục tiêu:
   - plastic  <- Plastic
   - paper    <- Paper, Cardboard
   - metal    <- Metal
   - glass    <- Glass
   - organic  <- Food Organics, Vegetation
   - other    <- Miscellaneous Trash, Textile Trash
2. Phân tầng (Stratified Sampling) theo 6 nhóm mục tiêu với tỷ lệ xấp xỉ 70% Train / 15% Val / 15% Test.
3. Cố định random_seed=42 để đảm bảo 100% tính tái lập (reproducibility).
4. Tạo manifest (CSV, JSON) chứa đường dẫn tương đối và nhãn, lưu tại data/processed/.
5. Không sao chép ảnh hay chỉnh sửa thư mục gốc data/raw/realwaste.
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Cấu hình mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đường dẫn thư mục dự án
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Bảng ánh xạ nhãn chuẩn từ 9 danh mục gốc sang 6 nhóm mục tiêu
LABEL_MAPPING = {
    "Plastic": "plastic",
    "Paper": "paper",
    "Cardboard": "paper",
    "Metal": "metal",
    "Glass": "glass",
    "Food Organics": "organic",
    "Vegetation": "organic",
    "Miscellaneous Trash": "other",
    "Textile Trash": "other"
}

TARGET_CLASSES = ["glass", "metal", "organic", "other", "paper", "plastic"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(sorted(TARGET_CLASSES))}
IDX_TO_CLASS = {str(idx): cls_name for cls_name, idx in CLASS_TO_IDX.items()}

EXPECTED_CATEGORIES = sorted(list(LABEL_MAPPING.keys()))
EXPECTED_TOTAL_IMAGES = 4752


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Phân chia tập dữ liệu RealWaste theo phương pháp phân tầng (Stratified Split)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Thư mục chứa dữ liệu RealWaste đã giải nén (Mặc định: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thư mục lưu trữ manifest kết quả phân chia (Mặc định: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để đảm bảo tính tái lập (Mặc định: 42)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Tỷ lệ tập Train (Mặc định: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Tỷ lệ tập Validation (Mặc định: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Tỷ lệ tập Test (Mặc định: 0.15)"
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        default=True,
        help="Kiểm tra tính toàn vẹn từng ảnh trước khi phân chia (Mặc định: True)"
    )
    return parser.parse_args()


def scan_dataset(data_dir: Path, verify_images: bool = True):
    """
    Quét và kiểm tra tính toàn vẹn của tập dữ liệu RealWaste.
    Xác thực tất cả 9 danh mục gốc, định dạng ảnh, tính đọc được của file.
    """
    print(f"\n[1/4] Đang quét và kiểm tra tập dữ liệu tại: {data_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy thư mục dữ liệu tại: {data_dir}")

    found_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    missing_dirs = set(EXPECTED_CATEGORIES) - set(found_dirs)
    unexpected_dirs = set(found_dirs) - set(EXPECTED_CATEGORIES)

    if missing_dirs:
        raise ValueError(f"Lỗi: Thiếu các thư mục danh mục gốc: {sorted(list(missing_dirs))}")
    if unexpected_dirs:
        raise ValueError(f"Cảnh báo: Phát hiện thư mục bất thường ngoài 9 danh mục chuẩn: {sorted(list(unexpected_dirs))}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    scanned_items = []
    category_counts = defaultdict(int)

    for cat_name in EXPECTED_CATEGORIES:
        cat_dir = data_dir / cat_name
        files = sorted(list(cat_dir.iterdir()), key=lambda x: x.name)

        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in valid_extensions:
                raise ValueError(f"Lỗi: Phát hiện tệp không phải ảnh: {file_path}")

            if file_path.stat().st_size == 0:
                raise ValueError(f"Lỗi: Tệp ảnh rỗng 0 byte: {file_path}")

            if verify_images:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    raise IOError(f"Lỗi: Tệp ảnh bị hỏng không đọc được: {file_path} ({e})")

            # Đường dẫn tương đối từ PROJECT_ROOT
            try:
                rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_path = file_path.as_posix()

            target_class = LABEL_MAPPING[cat_name]
            scanned_items.append({
                "image_path": rel_path,
                "original_category": cat_name,
                "target_class": target_class,
                "class_idx": CLASS_TO_IDX[target_class]
            })
            category_counts[cat_name] += 1

    total_scanned = len(scanned_items)
    print(f"  -> Quét thành công {total_scanned} ảnh từ 9 danh mục gốc.")
    if total_scanned != EXPECTED_TOTAL_IMAGES:
        print(f"  [LƯU Ý] Tổng số ảnh đo được ({total_scanned}) khác số chuẩn công bố ({EXPECTED_TOTAL_IMAGES}).")
    else:
        print(f"  [XÁC THỰC] Khớp chính xác 100% với số lượng {EXPECTED_TOTAL_IMAGES} ảnh.")

    return scanned_items, category_counts


def stratified_split(items, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Phân chia phân tầng (Stratified Split) theo 6 nhóm mục tiêu.
    Đảm bảo 100% tính tái lập bằng cách sort trước khi shuffle với seed cố định.
    """
    print(f"\n[2/4] Thực hiện phân chia phân tầng (Stratified Split) với Seed={seed}...")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-5:
        raise ValueError(f"Lỗi: Tổng tỷ lệ phải bằng 1.0 (Hiện tại: {ratio_sum})")

    # Gom nhóm theo 6 target classes
    by_target = defaultdict(list)
    for it in items:
        by_target[it["target_class"]].append(it)

    train_set = []
    val_set = []
    test_set = []

    rng = random.Random(seed)

    print(f"{'Target Class':<15} | {'Total':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 55)

    for target_class in sorted(by_target.keys()):
        class_items = by_target[target_class]
        # 1. Sắp xếp đường dẫn để tạo thứ tự khởi đầu xác định độc lập nền tảng
        class_items.sort(key=lambda x: x["image_path"])
        # 2. Xáo trộn ngẫu nhiên có kiểm soát bằng seed
        rng.shuffle(class_items)

        n_total = len(class_items)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = n_total - n_train - n_val

        # Đảm bảo mỗi tập đều có ít nhất 1 mẫu
        assert n_train > 0 and n_val > 0 and n_test > 0, f"Số lượng mẫu không đủ cho lớp {target_class}"

        train_part = class_items[:n_train]
        val_part = class_items[n_train:n_train + n_val]
        test_part = class_items[n_train + n_val:]

        train_set.extend(train_part)
        val_set.extend(val_part)
        test_set.extend(test_part)

        print(f"{target_class:<15} | {n_total:<8} | {n_train:<8} | {n_val:<8} | {n_test:<8}")

    print("-" * 55)
    print(f"{'TỔNG':<15} | {len(items):<8} | {len(train_set):<8} | {len(val_set):<8} | {len(test_set):<8}")

    return train_set, val_set, test_set


def verify_split_integrity(train_set, val_set, test_set, total_expected):
    """
    Kiểm tra toàn vẹn phân chia:
    - Không trùng lặp đường dẫn giữa các tập.
    - Không bỏ sót bất kỳ ảnh nào.
    - Tất cả 6 nhóm đều xuất hiện trong cả 3 tập.
    """
    print("\n[3/4] Kiểm tra các ràng buộc an toàn và tính toàn vẹn của phân chia...")
    train_paths = set(x["image_path"] for x in train_set)
    val_paths = set(x["image_path"] for x in val_set)
    test_paths = set(x["image_path"] for x in test_set)

    # 1. Kiểm tra kích thước tập
    total_assigned = len(train_paths) + len(val_paths) + len(test_paths)
    if total_assigned != total_expected:
        raise ValueError(f"Lỗi toàn vẹn: Tổng số ảnh gán ({total_assigned}) != Tổng ảnh quét ({total_expected})")

    # 2. Kiểm tra trùng lặp giữa các tập
    inter_train_val = train_paths & val_paths
    inter_train_test = train_paths & test_paths
    inter_val_test = val_paths & test_paths

    if inter_train_val:
        raise ValueError(f"Lỗi rò rỉ dữ liệu (Data Leakage): Trùng lặp giữa Train và Val ({len(inter_train_val)} ảnh)")
    if inter_train_test:
        raise ValueError(f"Lỗi rò rỉ dữ liệu (Data Leakage): Trùng lặp giữa Train và Test ({len(inter_train_test)} ảnh)")
    if inter_val_test:
        raise ValueError(f"Lỗi rò rỉ dữ liệu (Data Leakage): Trùng lặp giữa Val và Test ({len(inter_val_test)} ảnh)")

    # 3. Kiểm tra độ phủ của các lớp
    for split_name, split_set in [("Train", train_set), ("Validation", val_set), ("Test", test_set)]:
        classes_in_split = set(x["target_class"] for x in split_set)
        missing_classes = set(TARGET_CLASSES) - classes_in_split
        if missing_classes:
            raise ValueError(f"Lỗi: Tập {split_name} thiếu các lớp: {missing_classes}")

    print("  [PASS] 100% hình ảnh được gán vào duy nhất một tập.")
    print("  [PASS] Không có bất kỳ hình ảnh nào bị bỏ sót hay trùng lặp chéo.")
    print("  [PASS] Tất cả 6 lớp mục tiêu đều hiện diện đầy đủ trong cả Train, Val và Test.")


def write_manifests(output_dir: Path, train_set, val_set, test_set, seed, train_ratio, val_ratio, test_ratio):
    """
    Ghi các tệp manifest CSV, JSON và tệp tổng kết metadata vào output_dir.
    """
    print(f"\n[4/4] Đang ghi tệp manifest và metadata vào: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    fieldnames = ["image_path", "target_class", "class_idx", "original_category"]

    # 1. Ghi tệp CSV và JSON cho từng split
    for split_name, split_data in splits.items():
        # CSV
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)
        print(f"  -> Đã tạo: {csv_path.name} ({len(split_data)} dòng)")

        # JSON
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(split_data, f_json, indent=2, ensure_ascii=False)
        print(f"  -> Đã tạo: {json_path.name} ({len(split_data)} mục)")

    # 2. Tạo metadata tổng kết chi tiết (split_summary.json)
    total_count = len(train_set) + len(val_set) + len(test_set)
    summary_data = {
        "project": "AI-Based Waste Recognition and Recycling Recommendation System",
        "dataset_name": "RealWaste",
        "random_seed": seed,
        "target_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "total_images": total_count,
        "class_mapping": LABEL_MAPPING,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "splits": {}
    }

    for split_name, split_data in splits.items():
        class_dist = defaultdict(int)
        orig_dist = defaultdict(int)
        for item in split_data:
            class_dist[item["target_class"]] += 1
            orig_dist[item["original_category"]] += 1

        summary_data["splits"][split_name] = {
            "count": len(split_data),
            "percentage": round(len(split_data) / total_count * 100, 2),
            "class_distribution": dict(sorted(class_dist.items())),
            "original_category_distribution": dict(sorted(orig_dist.items()))
        }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2, ensure_ascii=False)
    print(f"  -> Đã tạo: {summary_path.name} (Tài liệu metadata tái lập)")


def main():
    args = parse_arguments()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("================================================================================")
    print("  QUY TRÌNH CHUẨN BỊ VÀ PHÂN CHIA TẬP DỮ LIỆU REALWASTE (PHASE 5.4.2A)")
    print("================================================================================")
    print(f"Thư mục dữ liệu gốc : {data_dir}")
    print(f"Thư mục xuất kết quả: {output_dir}")
    print(f"Random Seed         : {args.seed}")
    print(f"Tỷ lệ mục tiêu      : Train {args.train_ratio*100:.0f}% / Val {args.val_ratio*100:.0f}% / Test {args.test_ratio*100:.0f}%")
    print("--------------------------------------------------------------------------------")

    # 1. Quét dữ liệu
    items, category_counts = scan_dataset(data_dir, verify_images=args.verify_images)

    # 2. Phân tầng chia tập
    train_set, val_set, test_set = stratified_split(
        items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 3. Kiểm tra tính toàn vẹn
    verify_split_integrity(train_set, val_set, test_set, len(items))

    # 4. Ghi manifests
    write_manifests(
        output_dir,
        train_set,
        val_set,
        test_set,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    print("\n================================================================================")
    print("  CHUẨN BỊ DỮ LIỆU HOÀN TẤT THÀNH CÔNG VÀ SẴN SÀNG HUẤN LUYỆN!")
    print("================================================================================")


if __name__ == "__main__":
    main()
