import re
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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
    return g.compile(checkpointer=MemorySaver())


def sources_for_answer(answer: str, docs) -> str:
    used = []
    for n in re.findall(r"\[(\d+)\]", answer):
        n = int(n)
        if n not in used and 1 <= n <= len(docs):
            used.append(n)
    if not used:
        return "(no sources cited)"
    lines = []
    for n in used:
        d = docs[n - 1]
        src = d.metadata.get("source", "unknown")
        start = d.metadata.get("start_index", 0)
        lines.append(f"[{n}] {src} (offset {start})")
    return "\n".join(lines)