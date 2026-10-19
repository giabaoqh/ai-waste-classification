#!/usr/bin/env python3
"""
Script tải và chuẩn bị tập dữ liệu RealWaste an toàn và có tính tái lập.
Dự án: AI-Based Waste Recognition and Recycling Recommendation System.

THÔNG BÁO GIẤY PHÉP (LICENSE NOTICE):
--------------------------------------
Tập dữ liệu RealWaste có sự khác biệt về giấy phép giữa các kênh phát hành:
- UCI Machine Learning Repository phân phối theo giấy phép: CC BY 4.0 (Attribution).
- Kho lưu trữ của nhóm tác giả gốc (AILab-UOW) và IEEE DataPort ghi nhận: CC BY-NC-SA 4.0
  (Attribution-NonCommercial-ShareAlike - Phi thương mại & Chia sẻ tương tự).

Dự án này là Đồ án tốt nghiệp đại học phi lợi nhuận của sinh viên.
Để đảm bảo an toàn pháp lý tuyệt đối, dự án tuân thủ điều kiện chặt chẽ hơn:
CC BY-NC-SA 4.0 (chỉ sử dụng cho học thuật, ghi nhận tác giả đầy đủ, không bán thương mại).
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import urllib.parse
import hashlib
import zipfile
from pathlib import Path

# Cấu hình mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Cấu hình nguồn tải chính thức
OFFICIAL_DATASET_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
ALLOWED_HOST = "archive.ics.uci.edu"

# Thư mục đích mặc định (nằm ngoài git, được .gitignore bảo vệ)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "realwaste"


class SecurityError(Exception):
    """Lỗi bảo mật khi phát hiện hành vi không an toàn (Zip Slip, Path Traversal...)."""
    pass


class SecurityRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler ngăn chặn chuyển hướng sang giao thức không an toàn hoặc tên miền lạ."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Bảo mật: Từ chối chuyển hướng sang giao thức không phải HTTPS: {newurl}")
        if parsed.netloc.lower() != ALLOWED_HOST:
            raise ValueError(f"Bảo mật: Từ chối chuyển hướng sang tên miền ngoài danh sách cho phép: {parsed.netloc}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tải và giải nén an toàn tập dữ liệu RealWaste từ nguồn chính thức UCI ML Repository."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Thực hiện tải dữ liệu thực tế (nếu không có cờ này, script chạy ở chế độ Dry-run)."
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        default=False,
        help="Xác nhận đã đọc và chấp thuận các điều khoản giấy phép (CC BY-NC-SA 4.0 / CC BY 4.0)."
    )
    parser.add_argument(
        "--checksum",
        type=str,
        default=None,
        help="Mã băm SHA-256 tin cậy mong đợi (tùy chọn để kiểm tra tính toàn vẹn)."
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Thư mục lưu trữ dữ liệu (Mặc định: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Giải nén file ZIP sau khi tải thành công."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ghi đè file nếu file lưu trữ hoặc thư mục đích đã tồn tại."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=OFFICIAL_DATASET_URL,
        help=f"URL tải chính thức (Mặc định: {OFFICIAL_DATASET_URL})"
    )
    return parser.parse_args()


# Giới hạn an toàn phòng chống Zip Bomb và cạn kiệt tài nguyên
MAX_ENTRY_COUNT = 10_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
MAX_INDIVIDUAL_FILE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_COMPRESSION_RATIO = 100.0  # Tỷ lệ nén tối đa cho file > 1MB
MIN_FREE_DISK_BUFFER_BYTES = 1024 * 1024 * 1024  # 1 GB đệm trống


class ExtractionLimitError(Exception):
    """Lỗi vượt quá ngưỡng an toàn giải nén (Zip Bomb, số lượng file, kích thước...)."""
    pass


class DiskSpaceError(Exception):
    """Lỗi không đủ dung lượng đĩa trống an toàn."""
    pass


def is_safe_zip_path(target_dir: Path, member_name: str) -> bool:
    """Kiểm tra ngăn chặn lỗ hổng Zip Slip (Path Traversal), Windows drive paths, UNC paths, và absolute paths."""
    # Từ chối nếu là đường dẫn tuyệt đối hoặc chứa drive letter (C:\, D:\...)
    if os.path.isabs(member_name) or (len(member_name) > 1 and member_name[1] == ":"):
        return False
    # Từ chối nếu bắt đầu bằng gạch chéo hoặc UNC path (//server, \\server)
    if member_name.startswith("/") or member_name.startswith("\\"):
        return False
    # Từ chối nếu chứa thành phần thư mục cha '..' trong bất kỳ phân đoạn nào
    norm_parts = Path(member_name).parts
    if ".." in norm_parts:
        return False

    target_dir_resolved = target_dir.resolve()
    try:
        dest_path = (target_dir / member_name).resolve()
        # Đường dẫn giải nén phải là con của thư mục đích
        return target_dir_resolved in dest_path.parents or dest_path == target_dir_resolved
    except Exception:
        return False


def is_symlink_entry(member: zipfile.ZipInfo) -> bool:
    """Kiểm tra thuộc tính Unix mode bits để phát hiện POSIX Symlink trong file ZIP."""
    mode = member.external_attr >> 16
    return (mode & 0o170000) == 0o120000


def validate_and_extract_zip(zip_path: Path, extract_dir: Path, force: bool = False):
    """
    Kiểm tra toàn diện tính hợp lệ, bảo mật và giải nén nguyên tử (Atomic Extraction).
    Bảo toàn 100% dữ liệu cũ nếu quá trình ghi đè hoặc đổi tên thất bại ở bất kỳ giai đoạn nào.
    Không bao giờ để lại thư mục giải nén dở dang khi gặp lỗi.
    Không bao giờ tự ý ghi đè thư mục đích đã có dữ liệu nếu không có cờ force.
    """
    import shutil
    import uuid

    print(f"\n[KIỂM TRA ZIP] Đang kiểm tra tính hợp lệ của file nén: {zip_path.name}...")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Lỗi: File {zip_path} không phải là file ZIP hợp lệ.")

    # 1. Kiểm tra tồn tại thư mục đích và bảo vệ dữ liệu cũ
    if extract_dir.exists() and any(extract_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Thư mục giải nén '{extract_dir}' đã tồn tại và chứa dữ liệu.\n"
                f"Để bảo vệ dữ liệu hiện có, script từ chối ghi đè.\n"
                f"Vui lòng chỉ định thêm cờ '--force' nếu bạn muốn ghi đè có chủ đích."
            )
        else:
            print(f"[CẢNH BÁO GHI ĐÈ] Thư mục '{extract_dir}' đã tồn tại và sẽ được bảo vệ an toàn trong quá trình giải nén thay thế.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_entries = len(members)
        print(f"[KIỂM TRA BẢO MẬT] Quét bảo mật toàn bộ {total_entries} tệp tin trong archive...")

        # 2. Kiểm tra số lượng entry tối đa (Chống Zip Bomb theo số lượng)
        if total_entries > MAX_ENTRY_COUNT:
            raise ExtractionLimitError(
                f"BẢO MẬT: File ZIP chứa {total_entries} entries, vượt quá ngưỡng an toàn ({MAX_ENTRY_COUNT} entries)."
            )

        # 3. Kiểm tra từng entry: Traversal, Symlink, Kích thước, Tỷ lệ nén
        total_uncompressed_bytes = 0
        for member in members:
            # Chặn Symlink
            if is_symlink_entry(member):
                raise SecurityError(
                    f"BẢO MẬT: Phát hiện symlink bị cấm trong file ZIP: {member.filename}"
                )

            # Chặn Path Traversal
            if not is_safe_zip_path(extract_dir, member.filename):
                raise SecurityError(
                    f"BẢO MẬT: Phát hiện đường dẫn không an toàn trong file ZIP (Zip Slip): {member.filename}"
                )

            # Kiểm tra kích thước file đơn lẻ
            if member.file_size > MAX_INDIVIDUAL_FILE_BYTES:
                raise ExtractionLimitError(
                    f"BẢO MẬT: Tệp tin '{member.filename}' có kích thước {member.file_size / (1024*1024):.1f} MB, vượt quá giới hạn ({MAX_INDIVIDUAL_FILE_BYTES / (1024*1024):.1f} MB)."
                )

            # Kiểm tra tỷ lệ nén bất thường hoặc nén vô hạn (Zip Bomb)
            if member.file_size > 1024 * 1024:
                if member.compress_size == 0:
                    raise ExtractionLimitError(
                        f"BẢO MẬT: Phát hiện tệp '{member.filename}' có kích thước {member.file_size / (1024*1024):.1f} MB nhưng kích thước nén là 0 (nghi vấn Zip Bomb)."
                    )
                ratio = member.file_size / member.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise ExtractionLimitError(
                        f"BẢO MẬT: Phát hiện tỷ lệ nén bất thường ({ratio:.1f}x) tại tệp '{member.filename}' (nghi vấn Zip Bomb)."
                    )

            total_uncompressed_bytes += member.file_size

        # 4. Kiểm tra tổng dung lượng giải nén
        if total_uncompressed_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ExtractionLimitError(
                f"BẢO MẬT: Tổng dung lượng giải nén {total_uncompressed_bytes / (1024*1024*1024):.2f} GB, vượt quá ngưỡng an toàn ({MAX_TOTAL_UNCOMPRESSED_BYTES / (1024*1024*1024):.1f} GB)."
            )

        # 5. Kiểm tra dung lượng đĩa trống khả dụng (Disk space check)
        parent_dir = extract_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        disk_free = shutil.disk_usage(parent_dir).free
        required_space = int(total_uncompressed_bytes * 1.2) + MIN_FREE_DISK_BUFFER_BYTES
        if disk_free < required_space:
            raise DiskSpaceError(
                f"LỖI Ổ ĐĨA: Ổ đĩa chỉ còn {disk_free / (1024*1024*1024):.2f} GB trống. Cần tối thiểu {required_space / (1024*1024*1024):.2f} GB để giải nén an toàn."
            )

        print(f"[KIỂM TRA BẢO MẬT] Tất cả {total_entries} entries hợp lệ. Tổng dung lượng: {total_uncompressed_bytes / (1024*1024):.1f} MB.")

        # 6. Kiểm tra tính toàn vẹn CRC32 của các header và payload (sau khi các ngưỡng an toàn đã được xác nhận)
        corrupt_file = zf.testzip()
        if corrupt_file is not None:
            raise ValueError(f"Lỗi: File ZIP bị hỏng tại tệp con: {corrupt_file}")
        print("[KIỂM TRA ZIP] Cấu trúc file ZIP hoàn toàn nguyên vẹn (CRC check passed).")

        # 7. Giải nén nguyên tử (Atomic Extraction vào thư mục tạm trên cùng filesystem)
        temp_extract_dir = parent_dir / f".tmp_extract_{uuid.uuid4().hex[:8]}"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        backup_dir = None
        try:
            print(f"[GIẢI NÉN NGUYÊN TỬ] Đang giải nén dữ liệu vào thư mục tạm: {temp_extract_dir.name}...")
            zf.extractall(path=temp_extract_dir)

            # Nếu có cờ force và extract_dir cũ đã tồn tại, bảo tồn an toàn bằng cách đổi tên tạm sang backup_dir
            if extract_dir.exists():
                backup_dir = parent_dir / f".backup_{uuid.uuid4().hex[:8]}"
                extract_dir.rename(backup_dir)

            # Đổi tên thư mục tạm thành thư mục đích chính thức
            temp_extract_dir.rename(extract_dir)

            # Sau khi đổi tên thành công 100%, mới dọn dẹp thư mục backup cũ
            if backup_dir and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            print(f"[GIẢI NÉN THÀNH CÔNG] Dữ liệu đã được lưu chính thức tại: {extract_dir}")
        except Exception as e:
            # Nếu xảy ra lỗi ở bất kỳ bước nào:
            # 1. Tự động phục hồi lại thư mục cũ nếu đã đổi tên sang backup_dir
            if backup_dir and backup_dir.exists() and not extract_dir.exists():
                try:
                    backup_dir.rename(extract_dir)
                    print("[PHỤC HỒI] Đã khôi phục lại thư mục dữ liệu cũ nguyên vẹn.")
                except Exception as restore_err:
                    print(f"[CẢNH BÁO PHỤC HỒI] Không thể tự động đổi tên lại thư mục cũ: {restore_err}. Thư mục cũ đang ở: {backup_dir}")

            # 2. Dọn dẹp thư mục tạm dở dang, không để lại rác
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            raise RuntimeError(f"Lỗi trong quá trình giải nén: {e}")




def download_stream(url: str, dest_path: Path, expected_sha256: str = None, timeout: int = 30):
    """Tải file theo luồng an toàn qua file tạm .part và tính toán SHA-256."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"Lỗi bảo mật: Chỉ cho phép tải qua giao thức HTTPS. URL được cung cấp: {url}")
    if parsed_url.netloc.lower() != ALLOWED_HOST:
        raise ValueError(f"Lỗi bảo mật: Tên miền tải '{parsed_url.netloc}' không khớp với máy chủ chính thức '{ALLOWED_HOST}'.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    print(f"[TẢI DỮ LIỆU] Nguồn: {url}")
    print(f"[TẢI DỮ LIỆU] File tạm: {part_path.name}")

    opener = urllib.request.build_opener(SecurityRedirectHandler())
    req = urllib.request.Request(url, headers={"User-Agent": "WasteClassificationResearch/1.0"})

    sha256_hash = hashlib.sha256()
    downloaded_bytes = 0

    try:
        with opener.open(req, timeout=timeout) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise urllib.error.HTTPError(url, status_code, "Lỗi phản hồi từ máy chủ", response.headers, None)

            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length and content_length.isdigit() else None
            
            total_str = f"{total_size / (1024*1024):.1f} MB" if total_size else "Không xác định"
            print(f"[TẢI DỮ LIỆU] Dung lượng tải về dự kiến: {total_str}")

            with open(part_path, "wb") as f_out:
                chunk_size = 1024 * 64  # 64 KB
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    sha256_hash.update(chunk)
                    downloaded_bytes += len(chunk)
                    
                    if total_size:
                        percent = (downloaded_bytes / total_size) * 100
                        print(f"\rTiến độ: {downloaded_bytes / (1024*1024):.1f} MB / {total_str} ({percent:.1f}%)", end="", flush=True)

        print(f"\n[TẢI DỮ LIỆU] Đã tải xong: {downloaded_bytes / (1024*1024):.1f} MB.")
    except Exception as e:
        if part_path.exists():
            part_path.unlink()
        raise RuntimeError(f"Lỗi trong quá trình truyền dữ liệu: {e}")

    calculated_sha256 = sha256_hash.hexdigest()
    print(f"[MÃ BĂM] SHA-256 thực tế của file tải về: {calculated_sha256}")

    if expected_sha256:
        if calculated_sha256.lower() != expected_sha256.strip().lower():
            if part_path.exists():
                part_path.unlink()
            raise ValueError(
                f"LỖI TOÀN VẸN: SHA-256 không khớp!\n"
                f"  Mong đợi: {expected_sha256}\n"
                f"  Thực tế : {calculated_sha256}"
            )
        print("[XÁC THỰC] Mã SHA-256 khớp chính xác với mã kiểm tra đã cung cấp!")
    else:
        print("[CẢNH BÁO TOÀN VẸN] Không có mã SHA-256 tin cậy nào được cấu hình từ trước.")
        print("  Quá trình kiểm tra tính toàn vẹn chưa hoàn tất đối chiếu.")
        print(f"  Hãy lưu lại mã băm này để làm căn cứ tái lập: {calculated_sha256}")

    # Đổi tên file tạm thành file chính thức sau khi kiểm tra xong
    if dest_path.exists():
        dest_path.unlink()
    part_path.rename(dest_path)
    print(f"[HOÀN THÀNH] File tải đã được lưu chính thức tại: {dest_path}")
    return dest_path


def main():
    args = parse_arguments()
    dest_dir = Path(args.dest_dir).resolve()
    zip_path = dest_dir / "realwaste.zip"
    if not zip_path.exists() and (dest_dir / "realwaste-main.zip").exists():
        zip_path = dest_dir / "realwaste-main.zip"

    print("================================================================================")
    print("  QUY TRÌNH CHUẨN BỊ VÀ TẢI DỮ LIỆU REALWASTE (REPRODUCIBLE WORKFLOW)")
    print("================================================================================")
    print(f"Nguồn chính thức: {args.url}")
    print(f"Thư mục lưu trữ : {dest_dir}")
    print("--------------------------------------------------------------------------------")
    print("LƯU Ý GIẤY PHÉP (LICENSE NOTICE):")
    print("- UCI ML Repository liệt kê giấy phép: CC BY 4.0.")
    print("- Nhóm tác giả gốc (AILab-UOW) và IEEE DataPort chỉ định giấy phép: CC BY-NC-SA 4.0.")
    print("- Dự án Đồ án tốt nghiệp này tuân thủ chuẩn phi thương mại và ghi nhận tác giả đầy đủ.")
    print("--------------------------------------------------------------------------------")

    # Kiểm tra xác nhận giấy phép
    if not args.accept_license:
        print("\n[CHƯA XÁC NHẬN GIẤY PHÉP]:")
        print("Để tiếp tục, bạn cần xác nhận việc nắm rõ giấy phép của tập dữ liệu bằng cách:")
        print("Thêm tham số '--accept-license' khi chạy lệnh.")
        print("\nVí dụ lệnh hợp lệ:")
        print(f"  python {Path(__file__).name} --download --accept-license")
        sys.exit(0)

    # Chế độ Dry-run mặc định
    if not args.download:
        print("\n[CHẾ ĐỘ THỬ NGHIỆM - DRY-RUN]:")
        print("Script đang chạy ở chế độ mô phỏng an toàn (Dry-run). KHÔNG có kết nối mạng hay file nào được tạo.")
        print(f"1. Kiểm tra cấu hình URL: {args.url} (Hợp lệ: Giao thức HTTPS, tên miền archive.ics.uci.edu)")
        print(f"2. Đường dẫn file nén dự kiến: {zip_path}")
        print(f"3. Thư mục đích: {dest_dir} (Nằm trong data/ được .gitignore bảo vệ)")
        print("\nĐể thực hiện tải dữ liệu thật sự, hãy chạy lệnh với cờ '--download':")
        print(f"  python {Path(__file__).name} --download --accept-license")
        print("================================================================================")
        return

    # Kiểm tra file đã tồn tại
    if zip_path.exists() and not args.force:
        print(f"\n[THÔNG BÁO] File lưu trữ {zip_path.name} đã tồn tại trong {dest_dir}.")
        print("Nếu muốn tải lại từ đầu, hãy chỉ định thêm cờ '--force'.")
    else:
        download_stream(args.url, zip_path, expected_sha256=args.checksum)

    if args.extract:
        extract_target = dest_dir / "extracted"
        validate_and_extract_zip(zip_path, extract_target, force=args.force)

    print("\n================================================================================")
    print("QUY TRÌNH KẾT THÚC THÀNH CÔNG")
    print("================================================================================")


if __name__ == "__main__":
    main()
