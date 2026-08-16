from langchain_core.documents import Document
from langchain_community.document_loaders import ConfluenceLoader

from ragbot.config import settings


def load_confluence() -> list[Document]:
    """Load pages from a Confluence space as Documents.

    Requires CONFLUENCE_URL / USERNAME / API_TOKEN / SPACE_KEY in settings.
    """
    if not (settings.confluence_url and settings.confluence_api_token
            and settings.confluence_space_key):
        raise ValueError(
            "Confluence is not configured — set CONFLUENCE_* variables in .env"
        )

    loader = ConfluenceLoader(
        url=settings.confluence_url,
        username=settings.confluence_username,
        api_key=settings.confluence_api_token,
        space_key=settings.confluence_space_key,
        include_attachments=False,   # skip PDFs/images inside pages for now
        limit=50,                    # pages per API request (pagination handled internally)
        keep_markdown_format=True,   # convert Confluence HTML -> markdown-ish text
    )
    docs = loader.load()

    # normalize metadata for citations, same shape as the local loader.
    # ConfluenceLoader puts the page URL in metadata["source"] and title in ["title"].
    for d in docs:
        title = d.metadata.get("title", "untitled")
        # keep a human-readable source label; page URL stays available too
        d.metadata["source"] = title
        d.metadata["origin"] = "confluence"

    return docs