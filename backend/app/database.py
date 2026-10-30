"""
Database Module for AI Waste Recognition System (Phase 3).

Provides persistent SQLite storage for prediction history with:
- Strict parameterized queries to prevent SQL injection.
- JSON serialization for complex nested structures (probabilities, top_predictions, recommendation, quality_assessment).
- Zero storage of image binaries by default to preserve privacy and minimize disk footprint.
- Robust pagination, class filtering, and ISO-8601 date range queries.
- Thread-safe connection context managers with WAL (Write-Ahead Logging) mode.
- Local deployment note: Local history is not user-isolated since authentication is out of scope.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("waste_recognition_database")

# Project root calculation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configurable database path via environment variable
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "waste_history.db"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))


def get_db_path() -> Path:
    """Trả về đường dẫn tệp cơ sở dữ liệu SQLite hiện tại."""
    return DB_PATH


@contextmanager
def get_db_connection(db_path: Optional[Path] = None):
    """
    Context manager cung cấp kết nối SQLite với cấu hình chuẩn:
    - row_factory = sqlite3.Row cho phép truy cập theo tên cột.
    - Tự động commit transaction khi hoàn thành và rollback khi có ngoại lệ.
    """
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        # Tối ưu hóa hiệu năng và độ an toàn đồng thời
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error during transaction: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Khởi tạo bảng prediction_history và các chỉ mục (indexes) nếu chưa tồn tại.
    Chạy tự động khi ứng dụng khởi động.
    """
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing SQLite database at: {target_path}")

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        filename TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        label_vi TEXT NOT NULL,
        confidence REAL NOT NULL,
        is_uncertain INTEGER NOT NULL DEFAULT 0,
        uncertainty_note TEXT,
        probabilities_json TEXT NOT NULL,
        top_predictions_json TEXT NOT NULL,
        quality_assessment_json TEXT,
        recommendation_json TEXT,
        image_width INTEGER,
        image_height INTEGER,
        image_format TEXT,
        image_size_bytes INTEGER
    );
    """

    create_index_created_at_sql = """
    CREATE INDEX IF NOT EXISTS idx_pred_history_created_at 
    ON prediction_history (created_at DESC);
    """

    create_index_class_sql = """
    CREATE INDEX IF NOT EXISTS idx_pred_history_class 
    ON prediction_history (predicted_class);
    """

    with get_db_connection(target_path) as conn:
        conn.execute(create_table_sql)
        conn.execute(create_index_created_at_sql)
        conn.execute(create_index_class_sql)

        # Migration: Bổ sung các cột feedback nếu chưa tồn tại
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(prediction_history);").fetchall()}
        if "feedback_is_correct" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN feedback_is_correct INTEGER DEFAULT NULL;")
        if "feedback_correct_class" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN feedback_correct_class TEXT DEFAULT NULL;")
        if "feedback_note" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN feedback_note TEXT DEFAULT NULL;")
        if "feedback_at" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN feedback_at TEXT DEFAULT NULL;")
        if "feedback_status" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN feedback_status TEXT DEFAULT NULL;")
        if "is_waste_detected" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN is_waste_detected INTEGER DEFAULT 1;")
        if "detection_status" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN detection_status TEXT DEFAULT 'valid_waste';")
        if "rejection_reason" not in existing_cols:
            conn.execute("ALTER TABLE prediction_history ADD COLUMN rejection_reason TEXT DEFAULT NULL;")

        # Smart City Litter Reports Table
        create_litter_reports_table_sql = """
        CREATE TABLE IF NOT EXISTS litter_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            address TEXT,
            description TEXT,
            image_base64 TEXT,
            detected_waste_classes TEXT,
            severity TEXT NOT NULL DEFAULT 'medium',
            status TEXT NOT NULL DEFAULT 'pending',
            resolved_at TEXT,
            resolved_note TEXT,
            upvotes INTEGER NOT NULL DEFAULT 1,
            reported_by TEXT DEFAULT 'Người dân'
        );
        """
        conn.execute(create_litter_reports_table_sql)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_litter_status ON litter_reports (status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_litter_created_at ON litter_reports (created_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_litter_severity ON litter_reports (severity);")

        # Migration: Bổ sung các cột thông tin nguồn tin tức & tiêu đề cho litter_reports nếu chưa tồn tại
        existing_litter_cols = {row["name"] for row in conn.execute("PRAGMA table_info(litter_reports);").fetchall()}
        if "source_type" not in existing_litter_cols:
            conn.execute("ALTER TABLE litter_reports ADD COLUMN source_type TEXT DEFAULT 'citizen';")
        if "source_name" not in existing_litter_cols:
            conn.execute("ALTER TABLE litter_reports ADD COLUMN source_name TEXT DEFAULT 'Người dân';")
        if "source_url" not in existing_litter_cols:
            conn.execute("ALTER TABLE litter_reports ADD COLUMN source_url TEXT DEFAULT NULL;")
        if "title" not in existing_litter_cols:
            conn.execute("ALTER TABLE litter_reports ADD COLUMN title TEXT DEFAULT NULL;")

        _seed_initial_litter_reports_if_empty(conn)

    logger.info("Database schema, indexes, and migrations initialized successfully.")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Chuyển đổi một dòng SQLite Row thành Dict với giải mã JSON cho các cột cấu trúc."""
    data = dict(row)

    # Chuyển đổi boolean is_uncertain
    data["is_uncertain"] = bool(data.get("is_uncertain", 0))

    # Chuyển đổi boolean feedback_is_correct
    if data.get("feedback_is_correct") is not None:
        data["feedback_is_correct"] = bool(data["feedback_is_correct"])
    else:
        data["feedback_is_correct"] = None

    # Parse JSON fields
    for json_col in ("probabilities_json", "top_predictions_json", "quality_assessment_json", "recommendation_json"):
        val = data.get(json_col)
        key_name = json_col.replace("_json", "")
        if val:
            try:
                data[key_name] = json.loads(val)
            except Exception as e:
                logger.warning(f"Failed to parse JSON for column {json_col} (ID {data.get('id')}): {e}")
                data[key_name] = None
        else:
            data[key_name] = None

    # Non-waste and OOD detection fields
    data["is_waste_detected"] = bool(data.get("is_waste_detected", 1)) if data.get("is_waste_detected") is not None else True
    data["detection_status"] = data.get("detection_status") or "valid_waste"
    data["rejection_reason"] = data.get("rejection_reason")

    # Tái tạo image_dimensions tiện lợi cho frontend
    if data.get("image_width") and data.get("image_height"):
        data["image_dimensions"] = {
            "width": data["image_width"],
            "height": data["image_height"]
        }
    else:
        data["image_dimensions"] = None

    return data


def save_prediction_record(
    record_data: Dict[str, Any],
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Lưu một bản ghi phân loại rác thải thành công vào cơ sở dữ liệu.
    TUYỆT ĐỐI KHÔNG LƯU DỮ LIỆU NHỊ PHÂN CỦA ẢNH (Image Binary) để bảo mật quyền riêng tư.
    """
    created_at = record_data.get("created_at") or datetime.now(timezone.utc).isoformat()
    filename = record_data.get("filename", "unknown.jpg")
    predicted_class = record_data.get("predicted_class", "other")
    label_vi = record_data.get("label_vi", "Rác khác")
    confidence = float(record_data.get("confidence", 0.0))
    is_uncertain = 1 if record_data.get("is_uncertain") else 0
    uncertainty_note = record_data.get("uncertainty_note")

    # JSON serializations
    probabilities_json = json.dumps(record_data.get("probabilities", {}), ensure_ascii=False)
    top_predictions_json = json.dumps(record_data.get("top_predictions", []), ensure_ascii=False)
    
    qa = record_data.get("quality_assessment")
    quality_json = json.dumps(qa if isinstance(qa, dict) else (qa.dict() if hasattr(qa, "dict") else None), ensure_ascii=False) if qa else None

    rec = record_data.get("recommendation")
    rec_json = json.dumps(rec if isinstance(rec, dict) else (rec.dict() if hasattr(rec, "dict") else None), ensure_ascii=False) if rec else None

    # Dimensions & file metadata
    dims = record_data.get("image_dimensions") or {}
    width = dims.get("width") if isinstance(dims, dict) else None
    height = dims.get("height") if isinstance(dims, dict) else None
    image_format = record_data.get("image_format")
    image_size_bytes = record_data.get("image_size_bytes")
    is_waste_detected = 0 if record_data.get("is_waste_detected") is False else 1
    detection_status = record_data.get("detection_status", "valid_waste")
    rejection_reason = record_data.get("rejection_reason")

    insert_sql = """
    INSERT INTO prediction_history (
        created_at, filename, predicted_class, label_vi, confidence,
        is_uncertain, uncertainty_note, probabilities_json, top_predictions_json,
        quality_assessment_json, recommendation_json, image_width, image_height,
        image_format, image_size_bytes, is_waste_detected, detection_status, rejection_reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            insert_sql,
            (
                created_at, filename, predicted_class, label_vi, confidence,
                is_uncertain, uncertainty_note, probabilities_json, top_predictions_json,
                quality_json, rec_json, width, height, image_format, image_size_bytes,
                is_waste_detected, detection_status, rejection_reason
            )
        )
        record_id = cursor.lastrowid

        # Lấy lại bản ghi vừa tạo
        cursor.execute("SELECT * FROM prediction_history WHERE id = ?;", (record_id,))
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(f"Failed to fetch newly created history record with ID {record_id}")
        return _row_to_dict(row)


def get_history_records(
    page: int = 1,
    page_size: int = 10,
    class_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Truy vấn danh sách lịch sử có phân trang và bộ lọc:
    - page: Số trang (1-indexed)
    - page_size: Kích thước trang (mặc định 10)
    - class_name: Lọc theo nhóm rác thải (vd: plastic, paper, metal, glass, organic, other)
    - start_date / end_date: Lọc theo khoảng thời gian (ISO-8601)
    Trả về: (danh sách bản ghi, tổng số bản ghi thỏa điều kiện)
    """
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size

    where_clauses = []
    params: List[Any] = []

    if class_name and class_name.strip():
        where_clauses.append("predicted_class = ?")
        params.append(class_name.strip().lower())

    if start_date and start_date.strip():
        # Đảm bảo so sánh chuỗi ISO-8601 (hoặc bắt đầu ngày)
        where_clauses.append("created_at >= ?")
        params.append(start_date.strip())

    if end_date and end_date.strip():
        where_clauses.append("created_at <= ?")
        # Nếu chỉ truyền ngày YYYY-MM-DD, thêm mốc cuối ngày
        val = end_date.strip()
        if len(val) == 10:
            val += "T23:59:59.999Z"
        params.append(val)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    count_sql = f"SELECT COUNT(*) FROM prediction_history {where_sql};"
    select_sql = f"""
    SELECT * FROM prediction_history 
    {where_sql}
    ORDER BY created_at DESC, id DESC
    LIMIT ? OFFSET ?;
    """

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # 1. Đếm tổng số
        cursor.execute(count_sql, params)
        total_count = cursor.fetchone()[0]

        # 2. Truy vấn danh sách bản ghi
        cursor.execute(select_sql, params + [page_size, offset])
        rows = cursor.fetchall()
        items = [_row_to_dict(r) for r in rows]

    return items, total_count


def get_history_record_by_id(
    record_id: int,
    db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Lấy chi tiết một bản ghi lịch sử theo ID."""
    select_sql = "SELECT * FROM prediction_history WHERE id = ?;"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(select_sql, (record_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_dict(row)
        return None


def delete_history_record_by_id(
    record_id: int,
    db_path: Optional[Path] = None
) -> bool:
    """Xóa một bản ghi lịch sử đơn lẻ. Trả về True nếu bản ghi tồn tại và đã được xóa."""
    delete_sql = "DELETE FROM prediction_history WHERE id = ?;"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(delete_sql, (record_id,))
        return cursor.rowcount > 0


def clear_all_history(
    db_path: Optional[Path] = None
) -> int:
    """Xóa toàn bộ lịch sử phân loại trong cơ sở dữ liệu. Trả về số lượng bản ghi đã xóa."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_history;")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM prediction_history;")
        # Thu dọn không gian trống (VACUUM thực thi ngoài transaction)
        return count


def get_history_statistics(
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Tổng hợp thống kê nhanh về các phân loại đã thực hiện."""
    sql_total = "SELECT COUNT(*) FROM prediction_history;"
    sql_class_counts = """
    SELECT predicted_class, COUNT(*) as count 
    FROM prediction_history 
    GROUP BY predicted_class;
    """
    sql_uncertain_count = "SELECT COUNT(*) FROM prediction_history WHERE is_uncertain = 1;"

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_total)
        total = cursor.fetchone()[0]

        cursor.execute(sql_uncertain_count)
        uncertain_count = cursor.fetchone()[0]

        cursor.execute(sql_class_counts)
        class_counts = {row["predicted_class"]: row["count"] for row in cursor.fetchall()}

    return {
        "total_predictions": total,
        "uncertain_predictions": uncertain_count,
        "class_counts": class_counts
    }


CLASS_VI_MAP = {
    "plastic": "Nhựa (Plastic)",
    "paper": "Giấy / Carton (Paper)",
    "metal": "Kim loại (Metal)",
    "glass": "Thủy tinh (Glass)",
    "organic": "Hữu cơ (Organic)",
    "other": "Khác / Hỗn hợp (Other)"
}


def get_analytics_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Tổng hợp dữ liệu phân tích chi tiết cho Dashboard:
    - Tổng số dự đoán thành công
    - Điểm tin cậy trung bình và tỷ lệ không chắc chắn (uncertain)
    - Phân bố theo từng nhóm rác thải (số lượng, tỷ lệ %, điểm tin cậy trung bình)
    - Phân bố độ tin cậy theo các dải (buckets: >=90%, 80-89%, 65-79%, <65%)
    - Biểu đồ xu hướng phân loại theo ngày (daily trends)
    - Thống kê chất lượng hình ảnh đầu vào
    - 5 bản ghi phân loại gần nhất
    - Bộ lọc ngày tháng an toàn (parameterized)
    """
    where_clauses = []
    params: List[Any] = []

    if start_date and isinstance(start_date, str) and start_date.strip():
        where_clauses.append("created_at >= ?")
        params.append(start_date.strip())

    if end_date and isinstance(end_date, str) and end_date.strip():
        where_clauses.append("created_at <= ?")
        val = end_date.strip()
        if len(val) == 10:
            val += "T23:59:59.999Z"
        params.append(val)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Thống kê tổng quan
        sql_summary = f"""
        SELECT 
            COUNT(*) as total,
            COALESCE(AVG(confidence), 0.0) as avg_conf,
            COALESCE(SUM(is_uncertain), 0) as uncertain_cnt
        FROM prediction_history {where_sql};
        """
        cursor.execute(sql_summary, params)
        sum_row = cursor.fetchone()
        total = sum_row["total"]
        avg_conf = round(float(sum_row["avg_conf"]), 4)
        uncertain_cnt = int(sum_row["uncertain_cnt"])
        uncertain_pct = round((uncertain_cnt / total * 100), 1) if total > 0 else 0.0

        # 2. Phân bố theo lớp rác thải
        sql_classes = f"""
        SELECT 
            predicted_class, 
            COUNT(*) as count, 
            AVG(confidence) as avg_conf
        FROM prediction_history {where_sql}
        GROUP BY predicted_class
        ORDER BY count DESC;
        """
        cursor.execute(sql_classes, params)
        raw_class_rows = cursor.fetchall()
        class_counts_map = {r["predicted_class"]: r for r in raw_class_rows}

        class_distribution = []
        for cls_name in ("plastic", "paper", "metal", "glass", "organic", "other"):
            row = class_counts_map.get(cls_name)
            cnt = row["count"] if row else 0
            c_avg = round(float(row["avg_conf"]), 4) if row else 0.0
            pct = round((cnt / total * 100), 1) if total > 0 else 0.0
            class_distribution.append({
                "class_name": cls_name,
                "label_vi": CLASS_VI_MAP.get(cls_name, cls_name),
                "count": cnt,
                "percentage": pct,
                "avg_confidence": c_avg
            })

        # Sắp xếp class_distribution giảm dần theo count
        class_distribution.sort(key=lambda x: x["count"], reverse=True)

        # 3. Phân bố dải tin cậy (Confidence buckets)
        sql_buckets = f"""
        SELECT 
            CASE 
                WHEN confidence >= 0.90 THEN 'very_high'
                WHEN confidence >= 0.80 THEN 'high'
                WHEN confidence >= 0.65 THEN 'medium'
                ELSE 'low'
            END AS bucket,
            COUNT(*) as count
        FROM prediction_history {where_sql}
        GROUP BY bucket;
        """
        cursor.execute(sql_buckets, params)
        bucket_counts = {r["bucket"]: r["count"] for r in cursor.fetchall()}

        confidence_buckets = [
            {
                "range_key": "very_high",
                "label": ">= 90% (Rất cao)",
                "count": bucket_counts.get("very_high", 0),
                "percentage": round(bucket_counts.get("very_high", 0) / total * 100, 1) if total > 0 else 0.0
            },
            {
                "range_key": "high",
                "label": "80% - 89% (Cao)",
                "count": bucket_counts.get("high", 0),
                "percentage": round(bucket_counts.get("high", 0) / total * 100, 1) if total > 0 else 0.0
            },
            {
                "range_key": "medium",
                "label": "65% - 79% (Trung bình)",
                "count": bucket_counts.get("medium", 0),
                "percentage": round(bucket_counts.get("medium", 0) / total * 100, 1) if total > 0 else 0.0
            },
            {
                "range_key": "low",
                "label": "< 65% (Cần kiểm tra)",
                "count": bucket_counts.get("low", 0),
                "percentage": round(bucket_counts.get("low", 0) / total * 100, 1) if total > 0 else 0.0
            }
        ]

        # 4. Xu hướng theo ngày & theo tuần (Timeline Trends)
        sql_daily_trend = f"""
        SELECT 
            substr(created_at, 1, 10) as date_str,
            COUNT(*) as count
        FROM prediction_history {where_sql}
        GROUP BY date_str
        ORDER BY date_str ASC;
        """
        cursor.execute(sql_daily_trend, params)
        daily_trends = [{"date": r["date_str"], "count": r["count"]} for r in cursor.fetchall()]

        sql_weekly_trend = f"""
        SELECT 
            strftime('%Y-W%W', created_at) as week_str,
            COUNT(*) as count
        FROM prediction_history {where_sql}
        GROUP BY week_str
        ORDER BY week_str ASC;
        """
        cursor.execute(sql_weekly_trend, params)
        weekly_trends = [{"week": r["week_str"], "count": r["count"]} for r in cursor.fetchall()]

        # 5. Thống kê chất lượng ảnh
        and_where_qa = f"AND quality_assessment_json LIKE '%\"has_warnings\": true%'" if where_sql else "WHERE quality_assessment_json LIKE '%\"has_warnings\": true%'"
        sql_qa_warns = f"SELECT COUNT(*) FROM prediction_history {where_sql} {and_where_qa};"
        cursor.execute(sql_qa_warns, params)
        warn_cnt = cursor.fetchone()[0]
        clean_cnt = max(0, total - warn_cnt)
        warn_rate = round((warn_cnt / total * 100), 1) if total > 0 else 0.0

        # 6. Loại rác được nhận diện nhiều nhất (Most Frequent Category)
        most_frequent_class = None
        if class_distribution and total > 0:
            top_cat = max(class_distribution, key=lambda x: x["count"])
            if top_cat["count"] > 0:
                most_frequent_class = top_cat

        # 7. Thống kê đánh giá phản hồi người dùng (Feedback & Uncertainty Metrics)
        sql_fb = f"""
        SELECT
            COUNT(CASE WHEN feedback_is_correct IS NOT NULL OR feedback_status IS NOT NULL THEN 1 END) as total_fb,
            SUM(CASE WHEN feedback_is_correct = 1 OR feedback_status = 'correct' THEN 1 ELSE 0 END) as agreed,
            SUM(CASE WHEN feedback_is_correct = 0 OR feedback_status = 'incorrect' THEN 1 ELSE 0 END) as disagreed,
            SUM(CASE WHEN feedback_status = 'uncertain' THEN 1 ELSE 0 END) as uncertain_fb
        FROM prediction_history {where_sql};
        """
        cursor.execute(sql_fb, params)
        fb_row = cursor.fetchone()
        total_fb = fb_row["total_fb"] or 0
        agreed_fb = fb_row["agreed"] or 0
        disagreed_fb = fb_row["disagreed"] or 0
        uncertain_fb = fb_row["uncertain_fb"] or 0
        fb_acc_rate = round((agreed_fb / (agreed_fb + disagreed_fb) * 100), 1) if (agreed_fb + disagreed_fb) > 0 else 0.0

        # 8. Danh sách các ca lỗi / không chắc chắn để nhóm phát triển cải thiện mô hình (Active Learning)
        where_errors = []
        if where_sql:
            where_errors.append(where_sql.replace("WHERE ", ""))
        where_errors.append("(feedback_is_correct = 0 OR feedback_status = 'incorrect' OR feedback_status = 'uncertain' OR is_uncertain = 1)")
        where_error_sql = f"WHERE {' AND '.join(where_errors)}"

        sql_errors = f"""
        SELECT * FROM prediction_history {where_error_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT 20;
        """
        cursor.execute(sql_errors, params)
        error_cases = [_row_to_dict(r) for r in cursor.fetchall()]

        # 9. Top 5 bản ghi mới nhất
        sql_recent = f"""
        SELECT * FROM prediction_history {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT 5;
        """
        cursor.execute(sql_recent, params)
        recent_records = [_row_to_dict(r) for r in cursor.fetchall()]

    return {
        "total_predictions": total,
        "average_confidence": avg_conf,
        "uncertain_count": uncertain_cnt,
        "uncertain_percentage": uncertain_pct,
        "most_frequent_class": most_frequent_class,
        "class_distribution": class_distribution,
        "confidence_buckets": confidence_buckets,
        "daily_trends": daily_trends,
        "weekly_trends": weekly_trends,
        "quality_metrics": {
            "images_with_warnings": warn_cnt,
            "clean_images_count": clean_cnt,
            "warning_rate_percentage": warn_rate
        },
        "feedback_metrics": {
            "total_feedbacks": total_fb,
            "agreed_count": agreed_fb,
            "disagreed_count": disagreed_fb,
            "uncertain_feedback_count": uncertain_fb,
            "accuracy_rate": fb_acc_rate
        },
        "error_cases": error_cases,
        "recent_records": recent_records,
        "filter_applied": {
            "start_date": start_date,
            "end_date": end_date
        }
    }


def save_user_feedback(
    record_id: int,
    is_correct: Optional[bool] = None,
    correct_class: Optional[str] = None,
    note: Optional[str] = None,
    feedback_status: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """
    Lưu đánh giá đúng/sai/không chắc chắn và nhãn điều chỉnh của người dùng vào bản ghi lịch sử.
    feedback_status có thể là: 'correct', 'incorrect', 'uncertain'.
    """
    target_path = db_path or DB_PATH
    now_iso = datetime.now(timezone.utc).isoformat()
    
    # Xác định status chuẩn hóa
    if feedback_status is None:
        if is_correct is True:
            feedback_status = "correct"
        elif is_correct is False:
            feedback_status = "incorrect"
        else:
            feedback_status = "uncertain"

    is_correct_val = 1 if feedback_status == "correct" else (0 if feedback_status == "incorrect" else None)

    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE prediction_history
            SET feedback_is_correct = ?,
                feedback_status = ?,
                feedback_correct_class = ?,
                feedback_note = ?,
                feedback_at = ?
            WHERE id = ?;
            """,
            (is_correct_val, feedback_status, correct_class, note, now_iso, record_id)
        )
        return cursor.rowcount > 0


def get_error_cases_for_export(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Truy vấn toàn bộ các ca lỗi hoặc không chắc chắn để nhóm phát triển tải về huấn luyện lại mô hình."""
    target_path = db_path or DB_PATH
    sql = """
    SELECT * FROM prediction_history
    WHERE feedback_is_correct = 0 
       OR feedback_status = 'incorrect' 
       OR feedback_status = 'uncertain' 
       OR is_uncertain = 1
    ORDER BY created_at DESC, id DESC;
    """
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        return [_row_to_dict(r) for r in cursor.fetchall()]


def get_all_history_for_export(
    class_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Truy vấn toàn bộ bản ghi theo bộ lọc để xuất CSV (không phân trang)."""
    target_path = db_path or DB_PATH
    where_clauses, params = [], []

    if class_name:
        where_clauses.append("predicted_class = ?")
        params.append(class_name)

    if start_date:
        if len(start_date) == 10:
            where_clauses.append("created_at >= ?")
            params.append(f"{start_date}T00:00:00")
        else:
            where_clauses.append("created_at >= ?")
            params.append(start_date)

    if end_date:
        if len(end_date) == 10:
            where_clauses.append("created_at <= ?")
            params.append(f"{end_date}T23:59:59")
        else:
            where_clauses.append("created_at <= ?")
            params.append(end_date)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
    SELECT * FROM prediction_history
    {where_sql}
    ORDER BY created_at DESC, id DESC;
    """

    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [_row_to_dict(r) for r in rows]




# ==============================================================================
# SMART CITY CROWDSOURCED LITTER MAP MODULE
# ==============================================================================

INITIAL_LITTER_REPORTS_SEED = [
    {
        "latitude": 21.0368,
        "longitude": 105.7832,
        "address": "Ngõ 165 Cầu Giấy, P. Dịch Vọng, Q. Cầu Giấy, Hà Nội",
        "description": "Tập kết bao tải rác thải nhựa và thùng xốp lấn chiếm lòng đường ngõ dân cư.",
        "detected_waste_classes": ["plastic", "other"],
        "severity": "medium",
        "status": "pending",
        "upvotes": 14,
        "reported_by": "Cư dân Cầu Giấy",
        "created_at": "2026-10-30T08:30:00"
    },
    {
        "latitude": 21.0285,
        "longitude": 105.8542,
        "address": "Quảng trường Đông Kinh Nghĩa Thục, P. Tràng Tiền, Q. Hoàn Kiếm, Hà Nội",
        "description": "Thùng rác công cộng quá tải sau lễ hội đi bộ, nhiều ly nhựa dùng một lần và lon kim loại.",
        "detected_waste_classes": ["plastic", "metal", "paper"],
        "severity": "medium",
        "status": "in_progress",
        "upvotes": 32,
        "reported_by": "Đội Tình nguyện Xanh",
        "created_at": "2026-10-29T21:15:00"
    },
    {
        "latitude": 21.0150,
        "longitude": 105.8280,
        "address": "Gần cổng trạm y tế, P. Trung Liệt, Q. Đống Đa, Hà Nội",
        "description": "CẢNH BÁO: Phát hiện túi rác y tế (khẩu trang và găng tay y tế) vứt cạnh gốc cây.",
        "detected_waste_classes": ["medical", "plastic"],
        "severity": "critical",
        "status": "pending",
        "upvotes": 58,
        "reported_by": "Người dân phản ánh",
        "created_at": "2026-10-30T14:40:00"
    },
    {
        "latitude": 10.7769,
        "longitude": 106.7009,
        "address": "Đường Nguyễn Huệ, P. Bến Nghé, Quận 1, TP. Hồ Chí Minh",
        "description": "Đống vỏ chai nước khoáng và ly nhựa rải rác sau sự kiện âm nhạc.",
        "detected_waste_classes": ["plastic"],
        "severity": "low",
        "status": "resolved",
        "resolved_at": "2026-10-28T06:00:00",
        "resolved_note": "Công ty Vệ sinh Môi trường Đô thị đã thu gom và rửa sạch mặt bằng.",
        "upvotes": 41,
        "reported_by": "Người dân Quận 1",
        "created_at": "2026-10-27T23:10:00"
    },
    {
        "latitude": 10.8015,
        "longitude": 106.7112,
        "address": "Chân cầu Thị Nghè, P. 19, Q. Bình Thạnh, TP. Hồ Chí Minh",
        "description": "Bãi rác tự phát ven bờ kênh, phát sinh mùi hôi do thức ăn thừa và túi nilông.",
        "detected_waste_classes": ["organic", "plastic"],
        "severity": "high",
        "status": "pending",
        "upvotes": 26,
        "reported_by": "Cư dân Bình Thạnh",
        "created_at": "2026-10-30T10:05:00"
    },
    {
        "latitude": 16.0678,
        "longitude": 108.2208,
        "address": "Đường Bạch Đằng ven sông Hàn, P. Hải Châu 1, Q. Hải Châu, Đà Nẵng",
        "description": "Thùng rác phân loại 3 ngăn bị tràn vỏ chai thủy tinh và vỏ lon nhôm.",
        "detected_waste_classes": ["glass", "metal"],
        "severity": "medium",
        "status": "resolved",
        "resolved_at": "2026-10-29T15:30:00",
        "resolved_note": "Đã điều phối xe thu gom phân loại chuyển về trung tâm tái chế Hải Châu.",
        "upvotes": 19,
        "reported_by": "Du khách Đà Nẵng",
        "created_at": "2026-10-29T11:20:00"
    }
]


def _seed_initial_litter_reports_if_empty(conn: sqlite3.Connection) -> None:
    """Tạo trước dữ liệu mẫu thực tế tại Hà Nội, TP.HCM, Đà Nẵng nếu bảng đang trống."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM litter_reports;")
    count = cursor.fetchone()["count"]
    if count == 0:
        logger.info("Seeding initial smart city litter hotspot reports...")
        sql = """
        INSERT INTO litter_reports (
            created_at, latitude, longitude, address, description,
            image_base64, detected_waste_classes, severity, status,
            resolved_at, resolved_note, upvotes, reported_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        for item in INITIAL_LITTER_REPORTS_SEED:
            classes_json = json.dumps(item.get("detected_waste_classes", []), ensure_ascii=False)
            cursor.execute(sql, (
                item["created_at"],
                item["latitude"],
                item["longitude"],
                item["address"],
                item["description"],
                None,
                classes_json,
                item["severity"],
                item["status"],
                item.get("resolved_at"),
                item.get("resolved_note"),
                item.get("upvotes", 1),
                item.get("reported_by", "Người dân")
            ))
        logger.info(f"Seeded {len(INITIAL_LITTER_REPORTS_SEED)} initial smart city litter hotspot reports.")


def calculate_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính khoảng cách thực địa (km) giữa hai điểm tọa độ GPS theo công thức Haversine."""
    import math
    R = 6371.0  # Bán kính Trái Đất theo km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def create_litter_report(report_data: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Tạo mới một bản ghi báo cáo điểm nóng rác thải vào cơ sở dữ liệu."""
    target_path = db_path or DB_PATH
    created_at = report_data.get("created_at") or datetime.now(timezone.utc).isoformat()
    classes_json = json.dumps(report_data.get("detected_waste_classes", []), ensure_ascii=False)

    sql = """
    INSERT INTO litter_reports (
        created_at, latitude, longitude, address, description,
        image_base64, detected_waste_classes, severity, status,
        resolved_at, resolved_note, upvotes, reported_by,
        source_type, source_name, source_url, title
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    params = (
        created_at,
        float(report_data["latitude"]),
        float(report_data["longitude"]),
        report_data.get("address", ""),
        report_data.get("description", ""),
        report_data.get("image_base64"),
        classes_json,
        report_data.get("severity", "medium"),
        report_data.get("status", "pending"),
        report_data.get("resolved_at"),
        report_data.get("resolved_note"),
        int(report_data.get("upvotes", 1)),
        report_data.get("reported_by", "Người dân"),
        report_data.get("source_type", "citizen"),
        report_data.get("source_name", report_data.get("reported_by", "Người dân")),
        report_data.get("source_url"),
        report_data.get("title")
    )
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.lastrowid


def get_litter_reports(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    source_type: Optional[str] = None,
    user_lat: Optional[float] = None,
    user_lng: Optional[float] = None,
    radius_km: Optional[float] = None,
    sort_by: Optional[str] = None,
    limit: int = 150,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Truy vấn danh sách điểm nóng rác thải theo bộ lọc trạng thái, nguồn tin, bán kính GPS và mức độ nghiêm trọng."""
    target_path = db_path or DB_PATH
    where_clauses, params = [], []

    if status and status != "all":
        where_clauses.append("status = ?")
        params.append(status)

    if severity and severity != "all":
        where_clauses.append("severity = ?")
        params.append(severity)

    if source_type and source_type != "all":
        where_clauses.append("source_type = ?")
        params.append(source_type)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
    SELECT * FROM litter_reports
    {where_sql}
    ORDER BY 
        CASE severity 
            WHEN 'critical' THEN 1 
            WHEN 'high' THEN 2 
            WHEN 'medium' THEN 3 
            ELSE 4 
        END,
        created_at DESC
    LIMIT ?;
    """
    params.append(limit)

    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            try:
                d["detected_waste_classes"] = json.loads(d.get("detected_waste_classes") or "[]")
            except Exception:
                d["detected_waste_classes"] = []
            
            # Đảm bảo các giá trị mặc định cho trường nguồn
            d.setdefault("source_type", "citizen")
            d.setdefault("source_name", d.get("reported_by") or "Người dân")
            d.setdefault("source_url", None)
            d.setdefault("title", None)

            # Tính toán khoảng cách nếu có GPS người dùng
            if user_lat is not None and user_lng is not None:
                d["distance_km"] = calculate_haversine_km(
                    user_lat, user_lng, float(d["latitude"]), float(d["longitude"])
                )
            else:
                d["distance_km"] = None

            # Lọc theo bán kính nếu yêu cầu
            if radius_km is not None and d["distance_km"] is not None:
                if d["distance_km"] > radius_km:
                    continue

            results.append(d)

        # Sắp xếp theo cự ly gần nhất nếu sort_by == 'distance'
        if sort_by == "distance" and user_lat is not None and user_lng is not None:
            results.sort(key=lambda item: item["distance_km"] if item["distance_km"] is not None else 999999.0)

        return results


def get_litter_report_by_id(report_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Lấy thông tin chi tiết một điểm nóng rác thải theo ID."""
    target_path = db_path or DB_PATH
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM litter_reports WHERE id = ?;", (report_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["detected_waste_classes"] = json.loads(d.get("detected_waste_classes") or "[]")
        except Exception:
            d["detected_waste_classes"] = []
        return d


def update_litter_report_status(
    report_id: int,
    status: str,
    resolved_note: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Cập nhật trạng thái dọn dẹp ('pending', 'in_progress', 'resolved')."""
    target_path = db_path or DB_PATH
    resolved_at = datetime.now(timezone.utc).isoformat() if status == "resolved" else None

    sql = """
    UPDATE litter_reports
    SET status = ?,
        resolved_at = CASE WHEN ? = 'resolved' THEN COALESCE(?, resolved_at) ELSE NULL END,
        resolved_note = COALESCE(?, resolved_note)
    WHERE id = ?;
    """
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (status, status, resolved_at, resolved_note, report_id))
        return cursor.rowcount > 0


def upvote_litter_report(report_id: int, db_path: Optional[Path] = None) -> int:
    """Tăng số lượt xác nhận (upvote) của người dân đối với điểm nóng rác thải."""
    target_path = db_path or DB_PATH
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE litter_reports SET upvotes = upvotes + 1 WHERE id = ?;", (report_id,))
        cursor.execute("SELECT upvotes FROM litter_reports WHERE id = ?;", (report_id,))
        row = cursor.fetchone()
        return row["upvotes"] if row else 1


def get_litter_reports_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Tổng hợp số liệu thống kê bản đồ rác đô thị thông minh."""
    target_path = db_path or DB_PATH
    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM litter_reports;")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as pending FROM litter_reports WHERE status = 'pending';")
        pending = cursor.fetchone()["pending"]

        cursor.execute("SELECT COUNT(*) as in_prog FROM litter_reports WHERE status = 'in_progress';")
        in_progress = cursor.fetchone()["in_prog"]

        cursor.execute("SELECT COUNT(*) as resolved FROM litter_reports WHERE status = 'resolved';")
        resolved = cursor.fetchone()["resolved"]

        cursor.execute("SELECT COUNT(*) as critical FROM litter_reports WHERE severity = 'critical';")
        critical = cursor.fetchone()["critical"]

        resolution_rate = round((resolved / total * 100), 1) if total > 0 else 0.0

        return {
            "total_reports": total,
            "pending_reports": pending,
            "in_progress_reports": in_progress,
            "resolved_reports": resolved,
            "critical_reports": critical,
            "resolution_rate": resolution_rate
        }
