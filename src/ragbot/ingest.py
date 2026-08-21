from dataclasses import dataclass

from langchain_core.documents import Document

from ragbot.connectors.local_folder import load_folder
from ragbot.connectors.confluence import load_confluence
from ragbot.chunking import split_documents
from ragbot.vectorstore import get_vectorstore
from ragbot.ids import compute_ids
from ragbot.config import settings

LOCAL = "local"
CONFLUENCE = "confluence"


def load_all_documents() -> tuple[list[Document], set[str]]:
    """Load every configured source.

    Also returns the set of origins that loaded *successfully*. index() needs it:
    a connector that raised is not the same thing as a source that went empty,
    and only the first must be allowed to delete chunks.
    """
    docs = load_folder(settings.docs_dir)
    loaded = {LOCAL}

    # add Confluence only if it's configured
    if settings.confluence_url and settings.confluence_space_key:
        try:
            conf = load_confluence()
            print(f"[confluence] loaded {len(conf)} pages")
            docs.extend(conf)
            loaded.add(CONFLUENCE)
        except Exception as e:
            print(f"[confluence] skipped: {e}")

    return docs, loaded


def build_chunks() -> tuple[list[Document], list[Document], set[str]]:
    docs, loaded = load_all_documents()
    chunks = split_documents(docs)
    return docs, chunks, loaded


@dataclass(frozen=True)
class IndexResult:
    """What one indexing run did. Returned so a UI can report it without
    scraping stdout — the console script still prints the same numbers."""

    documents: int
    chunks: int
    added: int
    removed: int
    kept: int              # orphans left alone because their source failed to load
    unavailable: list[str]  # the origins that failed, if any

    @property
    def total(self) -> int:
        return self.chunks + self.kept


def index() -> IndexResult:
    docs, chunks, loaded = build_chunks()
    ids = compute_ids(chunks)
    print(f"\nLoaded documents: {len(docs)}")
    print(f"Produced chunks:  {len(chunks)}")

    store = get_vectorstore()
    stored = store.get(include=["metadatas"])
    existing = set(stored["ids"])
    current = set(ids)

    to_write = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    if to_write:
        store.add_documents([c for _, c in to_write], ids=[i for i, _ in to_write])

    # Chunks in the store that this run did not produce. Delete one only if its
    # origin actually loaded — otherwise a failing connector would wipe its own
    # content out of the index and silently degrade every later answer.
    # Rows written before origins existed have no `origin`; back then the only
    # unconditional source was the local folder, so treat them as local.
    stale, kept = [], []
    for chunk_id, meta in zip(stored["ids"], stored["metadatas"]):
        if chunk_id in current:
            continue
        origin = (meta or {}).get("origin", LOCAL)
        (stale if origin in loaded else kept).append(chunk_id)

    if stale:
        store.delete(ids=stale)

    unavailable = sorted({CONFLUENCE, LOCAL} - loaded)

    print(f"added/updated: {len(to_write)} | removed: {len(stale)} | "
          f"total now: {len(current) + len(kept)}")
    if kept:
        print(f"kept {len(kept)} chunk(s) from unavailable source(s): "
              f"{', '.join(unavailable) or 'unknown'}")

    return IndexResult(
        documents=len(docs),
        chunks=len(current),
        added=len(to_write),
        removed=len(stale),
        kept=len(kept),
        unavailable=unavailable,
    )


if __name__ == "__main__":
    index()
