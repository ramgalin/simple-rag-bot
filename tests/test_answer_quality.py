"""Answer-side checks.

The free half runs against real retrieved documents but a synthetic answer, so
it needs no API key. The `llm` half actually calls the model and is skipped
unless RAGBOT_LLM_TESTS=1 — it costs money and is not deterministic, so it does
not belong in a default `pytest` run.
"""

import os

import pytest

from ragbot.citations import sources_for_answer

needs_llm = pytest.mark.skipif(
    os.environ.get("RAGBOT_LLM_TESTS") != "1",
    reason="set RAGBOT_LLM_TESTS=1 to run tests that call the model",
)


# ----------------------------------------------------------------- free checks

def test_citations_resolve_to_the_documents_that_were_retrieved(demo_store):
    docs = demo_store.similarity_search("что такое принцип полезности?", k=4)
    listed = sources_for_answer("Согласно [1], польза — это... а в [3] уточняется.", docs)

    assert listed.splitlines() == [
        f"[1] {docs[0].metadata['source']} (offset {docs[0].metadata['start_index']})",
        f"[3] {docs[2].metadata['source']} (offset {docs[2].metadata['start_index']})",
    ]


def test_a_citation_beyond_the_context_is_dropped_not_crashed(demo_store):
    """Models do invent [9] when the context only had four documents."""
    docs = demo_store.similarity_search("что такое абсурд?", k=4)
    assert sources_for_answer("Как сказано в [9].", docs) == "(no sources cited)"


def test_offsets_point_back_into_the_real_file(demo_store):
    """The offset in a citation must be a real position in the source document."""
    docs = demo_store.similarity_search("who wrote the Meditations?", k=4)
    for d in docs:
        assert d.metadata["start_index"] >= 0
        assert d.metadata["source"].endswith(".md")


# ------------------------------------------------------------ live LLM checks

IN_DOMAIN = "что стоики говорят о вещах, которые не в нашей власти?"
OUT_OF_DOMAIN = "как настроить ingress в kubernetes?"


@pytest.fixture(scope="module")
def answer(tmp_path_factory):
    from ragbot.config import settings

    # never write test conversations into the user's real checkpoint database
    original = settings.checkpoint_db
    settings.checkpoint_db = str(tmp_path_factory.mktemp("checkpoints") / "test.sqlite")

    from ragbot.graph import build_graph, sources_for_answer as fmt

    graph = build_graph()
    counter = iter(range(1000))

    def _ask(question: str) -> tuple[str, str]:
        # a fresh thread per question: no history, no cross-contamination
        config = {"configurable": {"thread_id": f"pytest-{next(counter)}"}}
        result = graph.invoke({"question": question}, config=config)
        return result["answer"], fmt(result["answer"], result["docs"])

    yield _ask
    settings.checkpoint_db = original


@needs_llm
def test_answers_an_in_domain_question_with_a_citation(answer):
    text, sources = answer(IN_DOMAIN)
    assert sources != "(no sources cited)", f"answered without citing anything:\n{text}"
    assert "stoicism.md" in sources, f"cited the wrong document:\n{sources}"


@needs_llm
def test_refuses_an_out_of_domain_question(answer):
    """The prompt says to admit ignorance rather than improvise.

    Asserted through citations rather than wording: the phrasing varies with the
    question's language, but a genuine refusal cites nothing.
    """
    text, sources = answer(OUT_OF_DOMAIN)
    assert sources == "(no sources cited)", (
        f"improvised an answer from irrelevant context instead of refusing:\n{text}"
    )


@needs_llm
def test_answers_in_the_language_of_the_question(answer):
    russian, _ = answer(IN_DOMAIN)
    english, _ = answer("what is the only genuine good according to the Stoics?")
    assert any("а" <= c <= "я" for c in russian.lower()), russian
    assert not any("а" <= c <= "я" for c in english.lower()), english
