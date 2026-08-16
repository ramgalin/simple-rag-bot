from ragbot.vectorstore import get_vectorstore
from ragbot.config import settings


def get_retriever():
    """Turn the Chroma store into a retriever (a Runnable: query -> list[Document])."""
    store = get_vectorstore()
    return store.as_retriever(search_kwargs={"k": settings.retriever_k})