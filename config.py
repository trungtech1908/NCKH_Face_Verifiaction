"""
Config — load từ .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Qdrant ──
QDRANT_URL        = os.getenv("URL_QDRANT", "")
QDRANT_API_KEY    = os.getenv("API_QDRANT", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "face_users")

# ── JWT ──
JWT_SECRET   = os.getenv("JWT_SECRET", "change-this-secret-in-production")
JWT_EXPIRE_H = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# ── Face thresholds ──
FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.45"))
FACE_VERIFICATION_THRESHOLD = 0.5

# ── Anti-spoof ──
ANTI_SPOOF_SCORE_THRESHOLD = float(os.getenv("ANTI_SPOOF_SCORE_THRESHOLD", "0.85"))
ANTI_SPOOF_CONSISTENT_FRAMES = int(os.getenv("ANTI_SPOOF_CONSISTENT_FRAMES", "3"))
ANTI_SPOOF_STATIC_THRESHOLD = float(os.getenv("ANTI_SPOOF_STATIC_THRESHOLD", "2.0"))
ANTI_SPOOF_STATIC_FRAMES = int(os.getenv("ANTI_SPOOF_STATIC_FRAMES", "5"))

# ── MySQL ──
MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER     = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "face_verification")

# ── Validation ──
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Thiếu URL_QDRANT hoặc API_QDRANT trong file .env")
