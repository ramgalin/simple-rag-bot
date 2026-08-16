from langchain_community.embeddings import FastEmbedEmbeddings

from ragbot.config import settings


def build_embeddings() -> FastEmbedEmbeddings:
    # FastEmbed runs an ONNX model locally on CPU — no torch, no network calls.
    # The model file is downloaded once to ~/.cache on first use.
    return FastEmbedEmbeddings(model_name=settings.embedding_model)