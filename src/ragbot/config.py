from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"      # "anthropic" | "openai" | "langdock"
    llm_model: str = "claude-sonnet-4-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Langdock — OpenAI-compatible gateway: own key and base_url
    langdock_api_key: str | None = None
    langdock_base_url: str | None = None

    # documents and chunking
    docs_dir: str = "./docs"
    chunk_size: int = 800        # target chunk size in characters
    chunk_overlap: int = 100     # overlap between adjacent chunks

    # embeddings (local ONNX model via fastembed, 384-dim)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # vector store
    chroma_dir: str = "./data/chroma"
    collection_name: str = "ragbot"

    # retrieval
    retriever_k: int = 4     # how many chunks to pull per question

    # Confluence connector (optional)
    confluence_url: str | None = None
    confluence_username: str | None = None
    confluence_api_token: str | None = None
    confluence_space_key: str | None = None


settings = Settings()