from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def _load_one(path: Path) -> list[Document]:
    """Load a single file and return a list of Documents (one per page for PDFs)."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        # autodetect_encoding avoids cp1251/BOM issues in plain-text files
        return TextLoader(str(path), autodetect_encoding=True).load()

    if suffix == ".pdf":
        # PyPDFLoader returns one Document per page
        # and stores the page number in metadata["page"]
        return PyPDFLoader(str(path)).load()

    return []  # unknown type — silently skip


def load_folder(folder: str) -> list[Document]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root.resolve()}")

    docs: list[Document] = []
    files = sorted(p for p in root.rglob("*") if p.is_file())

    for path in files:
        try:
            loaded = _load_one(path)
        except Exception as e:
            # a single broken file must not kill the whole ingestion
            print(f"[skip] {path.name}: {e}")
            continue

        # normalize the source: relative path instead of an absolute one —
        # this is what will later show up in the citation
        rel = str(path.relative_to(root))
        for d in loaded:
            d.metadata["source"] = rel
            d.metadata["origin"] = "local"

        if loaded:
            docs.extend(loaded)
            print(f"[ok]   {rel} -> {len(loaded)} doc(s)")

    return docs