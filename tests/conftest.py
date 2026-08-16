import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


@pytest.fixture(scope="session")
def eval_set() -> dict:
    return json.loads((Path(__file__).parent / "eval_questions.json").read_text("utf-8"))


@pytest.fixture(scope="session")
def demo_store(tmp_path_factory):
    """A Chroma collection built from docs/ alone, in a throwaway directory.

    Deliberately NOT the project's own store: that one also holds Confluence
    content, depends on .env, and drifts as people edit pages — none of which
    belongs in a test. Building from docs/ keeps the evaluation reproducible for
    anyone who clones the repo, and needs no API key: embeddings run locally.
    """
    from langchain_chroma import Chroma

    from ragbot.chunking import split_documents
    from ragbot.connectors.local_folder import load_folder
    from ragbot.embeddings import build_embeddings
    from ragbot.ids import compute_ids

    chunks = split_documents(load_folder(str(DOCS)))
    store = Chroma(
        collection_name="eval",
        embedding_function=build_embeddings(),
        persist_directory=str(tmp_path_factory.mktemp("chroma-eval")),
    )
    store.add_documents(chunks, ids=compute_ids(chunks))
    return store


@pytest.fixture(scope="session")
def ranked_sources(demo_store):
    """question -> ranked list of source names, one entry per retrieved chunk."""

    def _search(question: str, k: int = 10) -> list[str]:
        return [d.metadata.get("source", "unknown")
                for d in demo_store.similarity_search(question, k=k)]

    return _search
