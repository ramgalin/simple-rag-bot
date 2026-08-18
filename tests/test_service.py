"""Service-layer tests.

The parts that need no model are tested directly; the parts that drive the graph
use a stub graph, so the contract between service and frontends is checked
without an API key. Live tests live in test_answer_quality.py.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from ragbot import service
from ragbot.citations import Citation

GENERATE = {"langgraph_node": "generate"}
CONDENSE = {"langgraph_node": "condense"}


@pytest.fixture
def docs():
    return [
        Document(page_content="Zeno taught in the Stoa Poikile.",
                 metadata={"source": "stoicism.md", "start_index": 0}),
        Document(page_content="Virtue is the only good.",
                 metadata={"source": "stoicism.md", "start_index": 800}),
    ]


def test_answer_exposes_structured_and_plain_citations(docs):
    answer = service.Answer(question="q", text="As [2] says.", docs=docs)

    assert answer.citations == [
        Citation(number=2, source="stoicism.md", start_index=800,
                 text="Virtue is the only good.")
    ]
    assert answer.sources_text == "[2] stoicism.md (offset 800)"


def test_answer_without_citations(docs):
    answer = service.Answer(question="q", text="I don't know.", docs=docs)
    assert answer.citations == []
    assert answer.sources_text == "(no sources cited)"


def test_citation_carries_the_chunk_text_for_a_side_panel(docs):
    """A GUI shows the retrieved text; it must not have to re-fetch it."""
    answer = service.Answer(question="q", text="See [1].", docs=docs)
    assert answer.citations[0].text == "Zeno taught in the Stoa Poikile."


class _StubGraph:
    """Stands in for the compiled graph: records calls, replays canned chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.configs = []

    def stream(self, payload, config, stream_mode):
        self.configs.append((payload, config, stream_mode))
        yield from self.chunks


@pytest.fixture
def stub(monkeypatch):
    def _install(chunks):
        g = _StubGraph(chunks)
        monkeypatch.setattr(service, "graph", lambda: g)
        return g
    return _install


def test_stream_answer_drops_tokens_from_other_nodes(stub):
    """condense also calls the LLM — its tokens must never reach a frontend."""
    stub([
        (AIMessageChunk(content="Где "), CONDENSE),
        (AIMessageChunk(content="он преподавал?"), CONDENSE),
        (AIMessageChunk(content="Зенон "), GENERATE),
        (AIMessageChunk(content="учил в Стое."), GENERATE),
    ])
    assert "".join(service.stream_answer("q", "s")) == "Зенон учил в Стое."


def test_stream_answer_drops_the_messages_written_to_state(stub):
    """`stream_mode="messages"` carries two different things.

    Besides the model's token chunks it also replays the finished messages the
    node appends to the state. Both arrive tagged with node 'generate', so
    filtering by node alone emits the answer twice with the question in between
    — which is exactly what happened before this was filtered by type.
    """
    stub([
        (AIMessageChunk(content="Зенон "), GENERATE),
        (AIMessageChunk(content="учил в Стое."), GENERATE),
        (HumanMessage(content="кто основал стоицизм?"), GENERATE),
        (AIMessage(content="Зенон учил в Стое."), GENERATE),
    ])
    assert "".join(service.stream_answer("q", "s")) == "Зенон учил в Стое."


def test_stream_answer_skips_empty_chunks(stub):
    stub([
        (AIMessageChunk(content=""), GENERATE),
        (AIMessageChunk(content="текст"), GENERATE),
    ])
    assert list(service.stream_answer("q", "s")) == ["текст"]


def test_session_becomes_the_thread_id(stub):
    g = stub([(AIMessageChunk(content="hi"), GENERATE)])
    list(service.stream_answer("вопрос", "my-chat"))

    payload, config, mode = g.configs[0]
    assert config == {"configurable": {"thread_id": "my-chat"}}
    assert payload == {"question": "вопрос"}
    assert mode == "messages"


def test_questions_are_sanitised_before_reaching_the_graph(stub):
    """A lone surrogate from stdin must not get past the service layer."""
    g = stub([(AIMessageChunk(content="ok"), GENERATE)])
    broken = "  вопрос" + b"\xd1".decode("utf-8", "surrogateescape") + "  "

    list(service.stream_answer(broken, "s"))

    payload, _, _ = g.configs[0]
    assert payload["question"] == "вопрос?"
    payload["question"].encode("utf-8")   # must not raise
