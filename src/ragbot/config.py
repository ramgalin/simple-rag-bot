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
    # Multilingual on purpose: questions get asked in Russian about English pages.
    # An -en- model scores "амбисоник" against "ambisonics" barely above noise.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # vector store
    chroma_dir: str = "./data/chroma"
    collection_name: str = "ragbot"

    # retrieval
    retriever_k: int = 8     # how many chunks to pull per question

    # conversation memory — LangGraph checkpointer, one row per (thread_id, step)
    checkpoint_db: str = "./data/checkpoints.sqlite"

    # web UI history (Chainlit's own store; ids match the checkpointer's thread ids)
    chainlit_db: str = "./data/chainlit.sqlite"

    # web UI login. No defaults on purpose: shipping a default password is worse
    # than refusing to start, and the sidebar needs an authenticated user.
    chainlit_user: str | None = None
    chainlit_password: str | None = None

    # Languages the answer-language detector chooses between. Keep this to the
    # languages you actually use: across all 75 lingua knows, short Cyrillic
    # questions get mistaken for other Slavic languages. Edit via DETECT_LANGUAGES
    # in .env — no code change needed.
    detect_languages: str = "English,Russian"

    # Confluence connector (optional)
    confluence_url: str | None = None
    confluence_username: str | None = None
    confluence_api_token: str | None = None
    confluence_space_key: str | None = None


settings = Settings()