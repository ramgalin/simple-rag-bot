import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from ragbot.citations import sources_for_answer
from ragbot.config import settings
from ragbot.retriever import get_retriever
from ragbot.llm import build_llm
from ragbot.prompts import CONDENSE_PROMPT, RAG_PROMPT
from ragbot.lang import detect_language


class State(TypedDict):
    # add_messages appends instead of overwriting — this is the running history
    messages: Annotated[list[BaseMessage], add_messages]
    question: str          # the latest raw user question
    standalone: str        # question after condensation
    docs: list[Document]   # retrieved chunks
    answer: str            # final answer text


_model = build_llm()
_retriever = get_retriever()


def condense(state: State) -> dict:
    """Rewrite the question into a standalone one, using history.
    Skipped-effect on first turn: with empty history the LLM returns it as-is."""
    history = state["messages"]
    if not history:
        return {"standalone": state["question"]}

    prompt = CONDENSE_PROMPT.invoke(
        {"chat_history": history, "question": state["question"]}
    )
    rewritten = _model.invoke(prompt).content.strip()
    return {"standalone": rewritten}


def retrieve(state: State) -> dict:
    docs = _retriever.invoke(state["standalone"])
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
    answer = _model.invoke(prompt).content
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
    _retriever.invoke("warmup")


def _checkpointer() -> SqliteSaver:
    """SQLite-backed checkpointer — conversations survive a restart."""
    path = Path(settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may touch the connection from a worker thread
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_graph():
    g = StateGraph(State)
    g.add_node("condense", condense)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.add_edge(START, "condense")
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    # checkpointer persists State per thread_id -> this IS the memory
    return g.compile(checkpointer=_checkpointer())


# re-exported so `from ragbot.graph import sources_for_answer` keeps working;
# it lives in citations.py because importing this module boots the LLM and Chroma
__all__ = ["State", "build_graph", "warmup", "sources_for_answer"]