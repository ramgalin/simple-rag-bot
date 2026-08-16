import warnings

from langchain_community.embeddings import FastEmbedEmbeddings

from ragbot.config import settings

# fastembed announces the CLS -> mean pooling switch for the multilingual MiniLM on
# every startup. The change is expected and retrieval quality was measured on it
# (see CLAUDE.md); do not pin fastembed back to 0.5.1 to silence it.
warnings.filterwarnings("ignore", message=".*now uses mean pooling.*")


def build_embeddings() -> FastEmbedEmbeddings:
    # FastEmbed runs an ONNX model locally on CPU — no torch, no network calls.
    # The model file is downloaded once to ~/.cache on first use.
    return FastEmbedEmbeddings(model_name=settings.embedding_model)