from langchain_chroma import Chroma

from ragbot.config import settings
from ragbot.embeddings import build_embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection on disk."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=build_embeddings(),
        persist_directory=settings.chroma_dir,   # this is what makes it survive restarts
    )