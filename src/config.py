import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)

for key in REQUIRED_ENV_VARS:
    if not os.getenv(key):
        raise RuntimeError(f"Environment variable {key} is not set")

PDF_PATH = os.getenv("PDF_PATH")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
