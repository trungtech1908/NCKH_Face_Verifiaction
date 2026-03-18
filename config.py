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

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError(
        "Thiếu URL_QDRANT hoặc API_QDRANT trong file .env"
    )
