"""sources_for_answer parses the [N] markers out of the model's answer, so the
printed sources are the ones actually cited rather than everything retrieved."""

from langchain_core.documents import Document

from ragbot.citations import sources_for_answer


def docs(n=3):
    return [
        Document(page_content="", metadata={"source": f"doc{i}.md", "start_index": i * 100})
        for i in range(1, n + 1)
    ]


def test_lists_only_cited_sources():
    out = sources_for_answer("Per [2] this holds.", docs())
    assert out == "[2] doc2.md (offset 200)"


def test_keeps_first_appearance_order_and_dedupes():
    out = sources_for_answer("[3] then [1], and [3] again.", docs())
    assert out.splitlines() == [
        "[3] doc3.md (offset 300)",
        "[1] doc1.md (offset 100)",
    ]


def test_ignores_out_of_range_citations():
    # a hallucinated [9] must not raise IndexError
    assert sources_for_answer("See [9] and [1].", docs()) == "[1] doc1.md (offset 100)"


def test_ignores_zero():
    assert sources_for_answer("See [0].", docs()) == "(no sources cited)"


def test_no_citations():
    assert sources_for_answer("I don't know.", docs()) == "(no sources cited)"


def test_falls_back_when_metadata_is_missing():
    out = sources_for_answer("[1]", [Document(page_content="", metadata={})])
    assert out == "[1] unknown (offset 0)"
