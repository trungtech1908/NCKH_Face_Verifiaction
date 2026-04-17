"""
Config — load từ .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL        = os.getenv("URL_QDRANT", "")
QDRANT_API_KEY    = os.getenv("API_QDRANT", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "face_users")

JWT_SECRET   = os.getenv("JWT_SECRET", "change-this-secret-in-production")
JWT_EXPIRE_H = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.45"))
FACE_VERIFICATION_THRESHOLD = 0.35

# Anti-spoof configuration
ANTI_SPOOF_SCORE_THRESHOLD = float(os.getenv("ANTI_SPOOF_SCORE_THRESHOLD", "0.85"))
ANTI_SPOOF_CONSISTENT_FRAMES = int(os.getenv("ANTI_SPOOF_CONSISTENT_FRAMES", "3"))
# If the face crop stays nearly identical across many frames (mean abs diff), mark as static spoof
ANTI_SPOOF_STATIC_THRESHOLD = float(os.getenv("ANTI_SPOOF_STATIC_THRESHOLD", "2.0"))
ANTI_SPOOF_STATIC_FRAMES = int(os.getenv("ANTI_SPOOF_STATIC_FRAMES", "5"))

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError(
        "Thiếu URL_QDRANT hoặc API_QDRANT trong file .env"
    )
