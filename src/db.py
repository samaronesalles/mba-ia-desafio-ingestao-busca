from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from config import COLLECTION_NAME, DATABASE_URL, OPENAI_EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
