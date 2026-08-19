"""Everything a frontend needs, so that frontends stay thin.

The CLI and any GUI must share one implementation of "ask a question": the
moment each of them assembles the graph, sanitises input and formats citations
on its own, the two drift apart and bugs get fixed in one but not the other.

What lives here and not in a frontend:

- the compiled graph, built **once** — `build_graph()` opens a fresh SQLite
  connection each call, which is fine for a process that starts one chat and
  ruinous for a web app that would do it per request;
- input sanitising, which used to sit in `cli.py` where a GUI could not reach it;
- session listing, because the sidebar of a chat UI is exactly the checkpointer's
  set of thread ids.

Both a blocking and a streaming entry point are provided. Chainlit and friends
are async, and a blocking `invoke` inside an async server stalls the event loop
for every connected user — so `astream_answer` exists for them, and `ask` stays
for the terminal.
"""

from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage

from ragbot.citations import Citation, cited, sources_for_answer
from ragbot.graph import (
    CONDENSE,
    GENERATE,
    RETRIEVE,
    build_async_graph,
    build_graph,
    warmup as _warmup,
)
from ragbot.textio import strip_surrogates

DEFAULT_SESSION = "default"

_graph = None


def _answer_tokens(chunk, meta) -> str:
    """The part of a `stream_mode="messages"` item that is answer text, if any.

    Two filters, both necessary:

    - by node, because `condense` also calls the LLM and its rewritten question
      would otherwise be printed as if it were the answer;
    - by type, because the stream carries not only the model's `AIMessageChunk`
      tokens but also the finished `HumanMessage`/`AIMessage` that the node
      appends to the state. Without this the answer is emitted twice, with the
      question wedged in between.
    """
    if meta.get("langgraph_node") != GENERATE:
        return ""
    if not isinstance(chunk, AIMessageChunk):
        return ""
    return chunk.content or ""


_async_graph = None


def graph():
    """The compiled graph for synchronous callers, built once and reused."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def agraph():
    """The compiled graph for asynchronous callers, built once and reused.

    A separate object because the checkpointers are not interchangeable:
    `SqliteSaver` raises NotImplementedError on every async method, so an async
    server driving the sync graph fails on the very first turn. Both write to
    the same SQLite file, so the conversation history is shared either way.
    """
    global _async_graph
    if _async_graph is None:
        _async_graph = await build_async_graph()
    return _async_graph


def warmup() -> None:
    """Load the embedding model before the first question arrives."""
    _warmup()


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    docs: list[Document]

    @property
    def citations(self) -> list[Citation]:
        """Structured sources — what a GUI needs to render a side panel."""
        return cited(self.text, self.docs)

    @property
    def sources_text(self) -> str:
        """Plain-text sources — what the terminal prints."""
        return sources_for_answer(self.text, self.docs)


def _config(session: str) -> dict:
    return {"configurable": {"thread_id": session}}


def _clean(question: str) -> str:
    return strip_surrogates(question.strip())


def ask(question: str, session: str = DEFAULT_SESSION) -> Answer:
    """Answer a question, blocking until the whole answer is ready."""
    question = _clean(question)
    result = graph().invoke({"question": question}, config=_config(session))
    return Answer(question=question, text=result["answer"], docs=result["docs"])


def stream_answer(question: str, session: str = DEFAULT_SESSION) -> Iterator[str]:
    """Yield answer tokens as they arrive (blocking iterator, for the CLI)."""
    question = _clean(question)
    for chunk, meta in graph().stream(
        {"question": question}, config=_config(session), stream_mode="messages"
    ):
        text = _answer_tokens(chunk, meta)
        if text:
            yield text


@dataclass(frozen=True)
class Rewritten:
    """condense finished: the question as it will actually be searched for."""

    question: str
    changed: bool


@dataclass(frozen=True)
class Retrieved:
    """retrieve finished: which documents will be in the model's context."""

    sources: list[str]      # distinct, in rank order
    chunks: int


@dataclass(frozen=True)
class Token:
    """One piece of the answer."""

    text: str


Event = Rewritten | Retrieved | Token


def _distinct(values: list[str]) -> list[str]:
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


async def astream_events(
    question: str, session: str = DEFAULT_SESSION
) -> AsyncIterator[Event]:
    """Stream both what the graph is doing and what it is answering.

    The gap between asking and the first token is several seconds — condensing
    and retrieval happen first — so a UI that only waits for tokens looks frozen.
    These events let it show the rewritten question and the documents found while
    the answer is still being generated.

    Ordering follows the graph: Rewritten, then Retrieved, then Tokens.
    """
    question = _clean(question)
    g = await agraph()

    async for mode, payload in g.astream(
        {"question": question},
        config=_config(session),
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            text = _answer_tokens(*payload)
            if text:
                yield Token(text)
            continue

        for node, update in payload.items():
            if node == CONDENSE:
                standalone = update.get("standalone", question)
                yield Rewritten(question=standalone, changed=standalone != question)
            elif node == RETRIEVE:
                docs = update.get("docs", [])
                yield Retrieved(
                    sources=_distinct([d.metadata.get("source", "unknown") for d in docs]),
                    chunks=len(docs),
                )


async def astream_answer(question: str, session: str = DEFAULT_SESSION) -> AsyncIterator[str]:
    """Answer tokens only — `astream_events` without the progress events.

    The graph's nodes are synchronous; LangGraph runs them in a worker thread,
    so this does not block the event loop.
    """
    async for event in astream_events(question, session):
        if isinstance(event, Token):
            yield event.text


def last_answer(session: str = DEFAULT_SESSION) -> Answer | None:
    """The answer this session finished with, read back from the checkpointer.

    Streaming yields text only; citations need the retrieved documents, and the
    graph already saved them. So a UI streams tokens, then calls this to render
    sources — instead of the stream having to carry two kinds of payload.
    """
    values = graph().get_state(_config(session)).values or {}
    if "answer" not in values:
        return None
    return Answer(
        question=values.get("question", ""),
        text=values["answer"],
        docs=values.get("docs", []),
    )


async def alast_answer(session: str = DEFAULT_SESSION) -> Answer | None:
    """Async counterpart of `last_answer`, for async frontends."""
    g = await agraph()
    values = (await g.aget_state(_config(session))).values or {}
    if "answer" not in values:
        return None
    return Answer(
        question=values.get("question", ""),
        text=values["answer"],
        docs=values.get("docs", []),
    )


def history(session: str = DEFAULT_SESSION) -> list[BaseMessage]:
    """Past turns of a conversation — what a UI replays when reopening a chat."""
    return (graph().get_state(_config(session)).values or {}).get("messages", [])


def sessions() -> list[str]:
    """Every stored conversation id — the sidebar list of a chat UI."""
    checkpointer = graph().checkpointer
    return sorted({
        c.config["configurable"]["thread_id"] for c in checkpointer.list(None)
    })


def delete_session(session: str) -> None:
    graph().checkpointer.delete_thread(session)
