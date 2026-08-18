import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ragbot.citations import sources_for_answer
from ragbot.config import settings
from ragbot.retriever import get_retriever
from ragbot.llm import build_llm
from ragbot.prompts import CONDENSE_PROMPT, RAG_PROMPT
from ragbot.lang import detect_language


# Node names are constants because streaming consumers filter on them: both
# condense and generate call the LLM, so a UI that does not filter would print
# the rewritten question along with the answer. A rename must not silently
# break that filter.
CONDENSE = "condense"
RETRIEVE = "retrieve"
GENERATE = "generate"


class State(TypedDict):
    # add_messages appends instead of overwriting — this is the running history
    messages: Annotated[list[BaseMessage], add_messages]
    question: str          # the latest raw user question
    standalone: str        # question after condensation
    docs: list[Document]   # retrieved chunks
    answer: str            # final answer text


# Built on first use, not at import. Importing this module used to read .env,
# open Chroma and load the embedding model as a side effect — which makes the
# module unimportable in a test that only needs the node names, and forces a web
# app to pay for it at import time rather than when it chooses to warm up.
_model = None
_retriever = None


def model():
    global _model
    if _model is None:
        _model = build_llm()
    return _model


def retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def condense(state: State) -> dict:
    """Rewrite the question into a standalone one, using history.
    Skipped-effect on first turn: with empty history the LLM returns it as-is."""
    history = state["messages"]
    if not history:
        return {"standalone": state["question"]}

    prompt = CONDENSE_PROMPT.invoke(
        {"chat_history": history, "question": state["question"]}
    )
    rewritten = model().invoke(prompt).content.strip()
    return {"standalone": rewritten}


def retrieve(state: State) -> dict:
    docs = retriever().invoke(state["standalone"])
    return {"docs": docs}


def _format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(blocks)


def generate(state: State) -> dict:
    prompt = RAG_PROMPT.invoke({
        "chat_history": state["messages"],
        "question": state["question"],
        "context": _format_docs(state["docs"]),
        "answer_language": detect_language(state["question"]),
    })
    answer = model().invoke(prompt).content
    # record this turn into history (question + answer)
    return {
        "answer": answer,
        "messages": [HumanMessage(state["question"]), AIMessage(answer)],
    }


def warmup() -> None:
    """Load the embedding model ahead of the first real question.

    Deliberately only touches the retriever: running the whole graph would also
    spend an LLM call and write a checkpoint on every startup.
    """
    retriever().invoke("warmup")


def _db_path() -> Path:
    path = Path(settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _checkpointer() -> SqliteSaver:
    """SQLite-backed checkpointer — conversations survive a restart."""
    # check_same_thread=False: LangGraph may touch the connection from a worker thread
    return SqliteSaver(sqlite3.connect(_db_path(), check_same_thread=False))


async def _async_checkpointer() -> AsyncSqliteSaver:
    """The same storage, reachable from async code.

    `SqliteSaver` raises NotImplementedError on every async method, so a graph
    compiled with it cannot be driven by `astream`/`ainvoke` at all — which is
    exactly what an async server does. Both savers use the same file and the
    same schema, so the two graphs share one history.
    """
    return AsyncSqliteSaver(await aiosqlite.connect(_db_path()))


def _wire() -> StateGraph:
    g = StateGraph(State)
    g.add_node(CONDENSE, condense)
    g.add_node(RETRIEVE, retrieve)
    g.add_node(GENERATE, generate)

    g.add_edge(START, CONDENSE)
    g.add_edge(CONDENSE, RETRIEVE)
    g.add_edge(RETRIEVE, GENERATE)
    g.add_edge(GENERATE, END)
    return g


def build_graph():
    """Graph for synchronous callers (the CLI)."""
    # checkpointer persists State per thread_id -> this IS the memory
    return _wire().compile(checkpointer=_checkpointer())


async def build_async_graph():
    """Graph for asynchronous callers (Chainlit, FastAPI)."""
    return _wire().compile(checkpointer=await _async_checkpointer())


# re-exported so `from ragbot.graph import sources_for_answer` keeps working;
# it lives in citations.py so that citation logic stays free of graph imports
__all__ = [
    "State", "build_graph", "build_async_graph", "warmup", "sources_for_answer",
    "CONDENSE", "RETRIEVE", "GENERATE",
]