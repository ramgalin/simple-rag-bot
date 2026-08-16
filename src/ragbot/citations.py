import re


def sources_for_answer(answer: str, docs) -> str:
    """List sources using the SAME numbers that appear in the answer.

    Only the citations the model actually wrote are listed — not every chunk
    that was retrieved.
    """
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
