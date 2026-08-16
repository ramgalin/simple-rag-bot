from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.config import settings


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,   # stores start_index in metadata — our precise anchor
    )
    return splitter.split_documents(docs)