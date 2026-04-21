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
    conn.commit()

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
