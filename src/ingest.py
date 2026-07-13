from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import COLLECTION_NAME, PDF_PATH
from db import get_vector_store


def ingest_pdf():
    docs = PyPDFLoader(str(PDF_PATH)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        raise RuntimeError("No splits found")

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    store = get_vector_store()
    store.add_documents(documents=enriched, ids=ids)

    print(f"Ingested {len(enriched)} documents into {COLLECTION_NAME}")
    print("Ingestion complete")


if __name__ == "__main__":
    ingest_pdf()
