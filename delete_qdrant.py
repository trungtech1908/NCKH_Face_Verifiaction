from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FilterSelector
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("URL_QDRANT")
api_key = os.getenv("API_QDRANT")
collection = os.getenv("QDRANT_COLLECTION")

client = QdrantClient(
    url=url,
    api_key=api_key
)

# ✅ Xoá toàn bộ points
client.delete(
    collection_name=collection,
    points_selector=FilterSelector(
        filter=Filter()
    )
)

print("Đã xoá toàn bộ dữ liệu trong collection")