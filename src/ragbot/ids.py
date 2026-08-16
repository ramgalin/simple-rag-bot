import hashlib

from langchain_core.documents import Document


def chunk_id(doc: Document) -> str:
    """Deterministic id for a chunk.

    Based on source + position + content, so the same chunk always maps to the
    same id (enabling upserts), while any edit to its text yields a new id.
    """
    source = doc.metadata.get("source", "unknown")
    start = doc.metadata.get("start_index", 0)
    payload = f"{source}:{start}:{doc.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_ids(chunks: list[Document]) -> list[str]:
    return [chunk_id(d) for d in chunks]