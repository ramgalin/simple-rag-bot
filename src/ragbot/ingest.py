from ragbot.connectors.local_folder import load_folder
from ragbot.connectors.confluence import load_confluence
from ragbot.chunking import split_documents
from ragbot.vectorstore import get_vectorstore
from ragbot.ids import compute_ids
from ragbot.config import settings


def load_all_documents():
    docs = load_folder(settings.docs_dir)

    # add Confluence only if it's configured
    if settings.confluence_url and settings.confluence_space_key:
        try:
            conf = load_confluence()
            print(f"[confluence] loaded {len(conf)} pages")
            docs.extend(conf)
        except Exception as e:
            print(f"[confluence] skipped: {e}")

    return docs


def build_chunks():
    docs = load_all_documents()
    chunks = split_documents(docs)
    return docs, chunks


def index() -> None:
    docs, chunks = build_chunks()
    ids = compute_ids(chunks)
    print(f"\nLoaded documents: {len(docs)}")
    print(f"Produced chunks:  {len(chunks)}")

    store = get_vectorstore()
    existing = set(store.get()["ids"])
    current = set(ids)

    to_write = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    if to_write:
        store.add_documents([c for _, c in to_write], ids=[i for i, _ in to_write])

    stale = list(existing - current)
    if stale:
        store.delete(ids=stale)

    print(f"added/updated: {len(to_write)} | removed: {len(stale)} | "
          f"total now: {len(current)}")


if __name__ == "__main__":
    index()