from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.config import settings
from ragbot.textio import strip_surrogates


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,   # stores start_index in metadata — our precise anchor
    )
    chunks = splitter.split_documents(docs)

    # Prepend the source (file name / Confluence page title) to every chunk.
    # Without it the title is only in metadata, so it is invisible to the
    # embedding — and a question like "what projects do we have" has nothing to
    # match against names like "l1ve-audio-sender". Done after splitting so the
    # header lands on every chunk, not just the first, and so start_index still
    # refers to the original document.
    for c in chunks:
        source = c.metadata.get("source")
        if source:
            c.page_content = f"{source}\n\n{c.page_content}"
        # TextLoader(autodetect_encoding=True) can hand back undecodable bytes as
        # surrogates; they would only fail much later, inside the LLM's HTTP client
        c.page_content = strip_surrogates(c.page_content)

    return chunks