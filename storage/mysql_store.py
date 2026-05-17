"""
MySQL Storage — User accounts
"""
import logging
from typing import Optional, List, Dict
import mysql.connector
from mysql.connector import Error
import config

logger = logging.getLogger(__name__)


def _get_connection():
    return mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE,
    )


def init_database():
    """Tạo database + table nếu chưa có, seed admin mặc định."""
    # Tạo database
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
                   f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()

    # Tạo table
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100) DEFAULT '',
            student_id VARCHAR(20) DEFAULT '',
            role ENUM('admin', 'user') DEFAULT 'user',
            face_registered BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # Lịch thi (exam schedule)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject VARCHAR(150) NOT NULL,
            exam_date DATE NOT NULL,
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            room VARCHAR(50) DEFAULT '',
            note VARCHAR(255) DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_exam_date (exam_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # Sinh viên trong mỗi lịch thi + mốc điểm danh + ảnh + điểm tự tin
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exam_students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            exam_id INT NOT NULL,
            user_id INT NOT NULL,
            attended_at DATETIME NULL,
            attendance_score FLOAT NULL,
            attendance_photo LONGBLOB NULL,
            UNIQUE KEY uniq_exam_user (exam_id, user_id),
            INDEX idx_exam (exam_id),
            INDEX idx_user (user_id),
            CONSTRAINT fk_es_exam FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE,
            CONSTRAINT fk_es_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()

    # Migration: bảng cũ (đã tạo trước khi có 2 cột này) → ALTER TABLE bổ sung.
    cursor.execute("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'exam_students'
    """)
    existing_cols = {row[0] for row in cursor.fetchall()}
    if "attendance_score" not in existing_cols:
        try:
            cursor.execute(
                "ALTER TABLE exam_students ADD COLUMN attendance_score FLOAT NULL AFTER attended_at"
            )
            conn.commit()
            logger.info("Added column exam_students.attendance_score")
        except Error as e:
            logger.warning("Could not add attendance_score column: %s", e)
    if "attendance_photo" not in existing_cols:
        try:
            cursor.execute(
                "ALTER TABLE exam_students ADD COLUMN attendance_photo LONGBLOB NULL"
            )
            conn.commit()
            logger.info("Added column exam_students.attendance_photo")
        except Error as e:
            logger.warning("Could not add attendance_photo column: %s", e)

    # Seed admin mặc định
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        from api.auth import hash_password
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, full_name, role) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("admin", "admin@system.local", hash_password("admin123"), "Administrator", "admin")
        )
        conn.commit()
        logger.info("Created default admin account (admin / admin123)")

    cursor.close()
    conn.close()
    logger.info("MySQL database initialized")


class MySQLUserStore:
    """CRUD operations for users table."""

    # ── CREATE ──

    def create_user(self, username: str, email: str, password_hash: str,
                    full_name: str = "", student_id: str = "",
                    role: str = "user") -> Optional[int]:
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name, student_id, role) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (username, email, password_hash, full_name, student_id, role)
            )
            conn.commit()
            user_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return user_id
        except Error as e:
            logger.error("create_user error: %s", e)
            return None

    # ── READ ──

    def _row_to_dict(self, row, cursor) -> Optional[Dict]:
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        d = dict(zip(columns, row))
        # Convert datetime to string
        for k in ("created_at", "updated_at"):
            if k in d and d[k] is not None:
                d[k] = d[k].strftime("%Y-%m-%dT%H:%M:%S")
        # Convert bool
        if "face_registered" in d:
            d["face_registered"] = bool(d["face_registered"])
        return d

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        result = self._row_to_dict(row, cursor)
        cursor.close()
        conn.close()
        return result

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        result = self._row_to_dict(row, cursor)
        cursor.close()
        conn.close()
        return result

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        result = self._row_to_dict(row, cursor)
        cursor.close()
        conn.close()
        return result

    def get_all_users(self, role: str = None) -> List[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        if role:
            cursor.execute("SELECT * FROM users WHERE role = %s ORDER BY id", (role,))
        else:
            cursor.execute("SELECT * FROM users ORDER BY id")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = self._row_to_dict(row, cursor)
            if d:
                results.append(d)
        cursor.close()
        conn.close()
        return results

    # ── UPDATE ──

    def update_user(self, user_id: int, **kwargs) -> bool:
        allowed = {"username", "email", "password_hash", "full_name", "student_id", "role", "face_registered"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = list(updates.values()) + [user_id]
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(f"UPDATE users SET {set_clause} WHERE id = %s", values)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Error as e:
            logger.error("update_user error: %s", e)
            return False

    def set_face_registered(self, user_id: int, registered: bool = True) -> bool:
        return self.update_user(user_id, face_registered=registered)

    # ── DELETE ──

    def delete_user(self, user_id: int) -> bool:
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Error as e:
            logger.error("delete_user error: %s", e)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# EXAM STORE — lịch thi + điểm danh
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_date(v) -> Optional[str]:
    if v is None:
        return None
    try:
        return v.strftime("%Y-%m-%d")
    except Exception:
        return str(v)


def _fmt_time(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if hasattr(v, "total_seconds"):
            s = int(v.total_seconds())
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
        return v.strftime("%H:%M")
    except Exception:
        return str(v)


def _fmt_dt(v) -> Optional[str]:
    if v is None:
        return None
    try:
        return v.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(v)


class ExamStore:
    """CRUD cho lịch thi (exams) và điểm danh (exam_students)."""

    def create_exam(self, *, subject: str, exam_date: str, start_time: str,
                    end_time: str, room: str = "", note: str = "",
                    student_ids: Optional[List[int]] = None) -> Optional[int]:
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO exams (subject, exam_date, start_time, end_time, room, note) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (subject, exam_date, start_time, end_time, room, note),
            )
            exam_id = cursor.lastrowid
            if student_ids:
                ids = list({int(x) for x in student_ids})
                cursor.executemany(
                    "INSERT IGNORE INTO exam_students (exam_id, user_id) VALUES (%s, %s)",
                    [(exam_id, uid) for uid in ids],
                )
            conn.commit()
            cursor.close()
            conn.close()
            return exam_id
        except Error as e:
            logger.error("create_exam error: %s", e)
            return None

    def list_exams(self) -> List[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.id, e.subject, e.exam_date, e.start_time, e.end_time,
                   e.room, e.note, e.created_at,
                   COUNT(es.id) AS total_students,
                   SUM(CASE WHEN es.attended_at IS NOT NULL THEN 1 ELSE 0 END) AS total_attended
            FROM exams e
            LEFT JOIN exam_students es ON es.exam_id = e.id
            GROUP BY e.id
            ORDER BY e.exam_date DESC, e.start_time DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "subject": r[1],
                "exam_date": _fmt_date(r[2]),
                "start_time": _fmt_time(r[3]),
                "end_time": _fmt_time(r[4]),
                "room": r[5] or "",
                "note": r[6] or "",
                "created_at": _fmt_dt(r[7]),
                "total_students": int(r[8] or 0),
                "total_attended": int(r[9] or 0),
            })
        return out

    def get_exam(self, exam_id: int) -> Optional[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, subject, exam_date, start_time, end_time, room, note, created_at "
            "FROM exams WHERE id = %s",
            (exam_id,),
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return None
        exam = {
            "id": row[0],
            "subject": row[1],
            "exam_date": _fmt_date(row[2]),
            "start_time": _fmt_time(row[3]),
            "end_time": _fmt_time(row[4]),
            "room": row[5] or "",
            "note": row[6] or "",
            "created_at": _fmt_dt(row[7]),
        }
        cursor.execute("""
            SELECT u.id, u.username, u.full_name, u.student_id, u.face_registered,
                   es.attended_at, es.attendance_score,
                   (es.attendance_photo IS NOT NULL) AS has_photo
            FROM exam_students es
            JOIN users u ON u.id = es.user_id
            WHERE es.exam_id = %s
            ORDER BY u.student_id, u.full_name, u.username
        """, (exam_id,))
        students = []
        for r in cursor.fetchall():
            students.append({
                "id": r[0],
                "username": r[1],
                "full_name": r[2] or "",
                "student_id": r[3] or "",
                "face_registered": bool(r[4]),
                "attended_at": _fmt_dt(r[5]),
                "attendance_score": float(r[6]) if r[6] is not None else None,
                "has_attendance_photo": bool(r[7]),
            })
        exam["students"] = students
        exam["total_students"] = len(students)
        exam["total_attended"] = sum(1 for s in students if s["attended_at"])
        cursor.close()
        conn.close()
        return exam

    def update_exam(self, exam_id: int, *, subject: Optional[str] = None,
                    exam_date: Optional[str] = None, start_time: Optional[str] = None,
                    end_time: Optional[str] = None, room: Optional[str] = None,
                    note: Optional[str] = None,
                    student_ids: Optional[List[int]] = None) -> bool:
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            fields = {
                "subject": subject, "exam_date": exam_date,
                "start_time": start_time, "end_time": end_time,
                "room": room, "note": note,
            }
            updates = {k: v for k, v in fields.items() if v is not None}
            if updates:
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                cursor.execute(
                    f"UPDATE exams SET {set_clause} WHERE id = %s",
                    list(updates.values()) + [exam_id],
                )
            if student_ids is not None:
                new_ids = {int(x) for x in student_ids}
                cursor.execute("SELECT user_id FROM exam_students WHERE exam_id = %s", (exam_id,))
                old_ids = {r[0] for r in cursor.fetchall()}
                to_add = new_ids - old_ids
                to_del = old_ids - new_ids
                if to_add:
                    cursor.executemany(
                        "INSERT IGNORE INTO exam_students (exam_id, user_id) VALUES (%s, %s)",
                        [(exam_id, uid) for uid in to_add],
                    )
                if to_del:
                    cursor.execute(
                        "DELETE FROM exam_students WHERE exam_id = %s AND user_id IN (%s)"
                        % (exam_id, ",".join(str(x) for x in to_del))
                    )
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Error as e:
            logger.error("update_exam error: %s", e)
            return False

    def find_room_conflict(self, *, room: str, exam_date: str,
                           start_time: str, end_time: str,
                           exclude_id: Optional[int] = None) -> Optional[Dict]:
        """
        Tìm lịch thi khác đã chiếm phòng `room` ngày `exam_date`
        có giờ giao với [start_time, end_time). Trả về dict {id, subject,
        start_time, end_time} nếu có xung đột, ngược lại None.
        """
        room = (room or "").strip()
        if not room:
            return None
        try:
            conn = _get_connection()
            cursor = conn.cursor(dictionary=True)
            sql = (
                "SELECT id, subject, start_time, end_time FROM exams "
                "WHERE room = %s AND exam_date = %s "
                "AND start_time < %s AND end_time > %s"
            )
            params = [room, exam_date, end_time, start_time]
            if exclude_id is not None:
                sql += " AND id <> %s"
                params.append(exclude_id)
            sql += " LIMIT 1"
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row:
                return None
            return {
                "id": row["id"],
                "subject": row["subject"],
                "start_time": _fmt_time(row["start_time"]),
                "end_time": _fmt_time(row["end_time"]),
            }
        except Error as e:
            logger.error("find_room_conflict error: %s", e)
            return None

    def delete_exam(self, exam_id: int) -> bool:
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM exams WHERE id = %s", (exam_id,))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Error as e:
            logger.error("delete_exam error: %s", e)
            return False

    def find_student_conflicts(self, *, student_ids: List[int], exam_date: str,
                               start_time: str, end_time: str,
                               exclude_exam_id: Optional[int] = None) -> List[Dict]:
        """
        Với danh sách student_ids sẽ được gán vào 1 lịch thi ngày `exam_date`
        giờ [start_time, end_time), tìm các sinh viên đã có mặt trong lịch thi
        khác có giờ chồng chéo.
        Trả về list {user_id, full_name, student_id, exam_id, subject, start_time, end_time}
        """
        if not student_ids:
            return []
        try:
            conn = _get_connection()
            cursor = conn.cursor(dictionary=True)
            placeholders = ",".join(["%s"] * len(student_ids))
            sql = (
                f"SELECT es.user_id, u.full_name, u.student_id, u.username, "
                f"       e.id AS exam_id, e.subject, e.start_time, e.end_time "
                f"FROM exam_students es "
                f"JOIN exams e ON e.id = es.exam_id "
                f"JOIN users u ON u.id = es.user_id "
                f"WHERE es.user_id IN ({placeholders}) "
                f"  AND e.exam_date = %s "
                f"  AND e.start_time < %s AND e.end_time > %s"
            )
            params: list = list(student_ids) + [exam_date, end_time, start_time]
            if exclude_exam_id is not None:
                sql += " AND e.id <> %s"
                params.append(exclude_exam_id)
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            out = []
            for r in rows:
                out.append({
                    "user_id": r["user_id"],
                    "full_name": r["full_name"] or "",
                    "student_id": r["student_id"] or "",
                    "username": r["username"] or "",
                    "exam_id": r["exam_id"],
                    "subject": r["subject"],
                    "start_time": _fmt_time(r["start_time"]),
                    "end_time": _fmt_time(r["end_time"]),
                })
            return out
        except Error as e:
            logger.error("find_student_conflicts error: %s", e)
            return []

    def list_exams_for_user(self, user_id: int) -> List[Dict]:
        """Danh sách lịch thi của 1 sinh viên kèm attended_at."""
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.id, e.subject, e.exam_date, e.start_time, e.end_time,
                       e.room, e.note, es.attended_at, es.attendance_score
                FROM exam_students es
                JOIN exams e ON e.id = es.exam_id
                WHERE es.user_id = %s
                ORDER BY e.exam_date DESC, e.start_time DESC
            """, (user_id,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            out = []
            for r in rows:
                out.append({
                    "id": r[0],
                    "subject": r[1],
                    "exam_date": _fmt_date(r[2]),
                    "start_time": _fmt_time(r[3]),
                    "end_time": _fmt_time(r[4]),
                    "room": r[5] or "",
                    "note": r[6] or "",
                    "attended_at": _fmt_dt(r[7]),
                    "attendance_score": float(r[8]) if r[8] is not None else None,
                })
            return out
        except Error as e:
            logger.error("list_exams_for_user error: %s", e)
            return []

    def unmark_attendance(self, exam_id: int, user_id: int) -> bool:
        """Huỷ điểm danh: xoá attended_at, attendance_score và attendance_photo.
        Trả True nếu có ít nhất một bản ghi bị thay đổi."""
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE exam_students "
                "SET attended_at = NULL, attendance_score = NULL, attendance_photo = NULL "
                "WHERE exam_id = %s AND user_id = %s",
                (exam_id, user_id),
            )
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Error as e:
            logger.error("unmark_attendance error: %s", e)
            return False

    def mark_attendance(
        self,
        exam_id: int,
        user_id: int,
        score: Optional[float] = None,
        photo_bytes: Optional[bytes] = None,
    ) -> Optional[Dict]:
        """
        Đánh dấu user_id có mặt ở exam_id. Chỉ set attended_at lần đầu
        (các lần sau không overwrite → giữ mốc xuất hiện đầu tiên).

        Khi `first_time=True` và caller có truyền `score`/`photo_bytes`, hai
        giá trị này cũng được lưu kèm. Khi mark thủ công không có ảnh thì cả
        2 trường vẫn NULL.

        Trả {attended_at, first_time, attendance_score, has_attendance_photo}
        hoặc None nếu sinh viên không thuộc lịch.
        """
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, attended_at, attendance_score, "
                "(attendance_photo IS NOT NULL) "
                "FROM exam_students WHERE exam_id = %s AND user_id = %s",
                (exam_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                conn.close()
                return None
            es_id, attended_at, cur_score, cur_has_photo = row
            first_time = attended_at is None
            if first_time:
                cursor.execute(
                    "UPDATE exam_students "
                    "SET attended_at = NOW(), "
                    "    attendance_score = %s, "
                    "    attendance_photo = %s "
                    "WHERE id = %s",
                    (
                        float(score) if score is not None else None,
                        photo_bytes if photo_bytes else None,
                        es_id,
                    ),
                )
                conn.commit()
                cursor.execute(
                    "SELECT attended_at, attendance_score, "
                    "(attendance_photo IS NOT NULL) "
                    "FROM exam_students WHERE id = %s",
                    (es_id,),
                )
                attended_at, cur_score, cur_has_photo = cursor.fetchone()
            cursor.close()
            conn.close()
            return {
                "attended_at": _fmt_dt(attended_at),
                "first_time": first_time,
                "attendance_score": float(cur_score) if cur_score is not None else None,
                "has_attendance_photo": bool(cur_has_photo),
            }
        except Error as e:
            logger.error("mark_attendance error: %s", e)
            return None

    def get_attendance_photo(self, exam_id: int, user_id: int) -> Optional[bytes]:
        """Trả bytes JPEG của ảnh điểm danh (nếu có), hoặc None."""
        try:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT attendance_photo FROM exam_students "
                "WHERE exam_id = %s AND user_id = %s",
                (exam_id, user_id),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row or row[0] is None:
                return None
            return bytes(row[0])
        except Error as e:
            logger.error("get_attendance_photo error: %s", e)
            return None
