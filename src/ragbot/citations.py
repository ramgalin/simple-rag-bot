import re
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class Citation:
    """One source the model actually referenced, with the chunk behind it."""

    number: int          # the [N] as written in the answer
    source: str          # file name or Confluence page title
    start_index: int     # character offset inside the original document
    text: str            # the retrieved chunk itself


def cited(answer: str, docs: list[Document]) -> list[Citation]:
    """Citations the answer actually uses, in order of first appearance.

    One parser, two renderings: the CLI prints `sources_for_answer` below the
    answer, a GUI wants `Citation.text` to show the chunk in a side panel. Both
    must agree about which sources were used, so neither re-parses `[N]`.

    Numbers the model invented (out of range) are dropped rather than raising —
    a hallucinated [9] against four documents is a normal occurrence.
    """
    out: list[Citation] = []
    seen: set[int] = set()
    for raw in re.findall(r"\[(\d+)\]", answer):
        number = int(raw)
        if number in seen or not (1 <= number <= len(docs)):
            continue
        seen.add(number)
        doc = docs[number - 1]
        out.append(
            Citation(
                number=number,
                source=doc.metadata.get("source", "unknown"),
                start_index=doc.metadata.get("start_index", 0),
                text=doc.page_content,
            )
        )
    return out


def sources_for_answer(answer: str, docs) -> str:
    """Plain-text rendering of `cited()`, as printed by the CLI."""
    citations = cited(answer, docs)
    if not citations:
        return "(no sources cited)"
    return "\n".join(
        f"[{c.number}] {c.source} (offset {c.start_index})" for c in citations
    )
