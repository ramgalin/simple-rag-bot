"""Retrieval quality metrics.

Kept separate from the tests so the metric implementations can themselves be
unit-tested — a silently wrong metric is worse than no metric.

`ranked` is always a list of *chunk* sources, in retrieval order and with
duplicates kept — position i is the source of the i-th retrieved chunk. Ranking
by chunk rather than by document is deliberate: `retriever_k` counts chunks, so
"did the right document reach the context" is a question about chunk positions.
Collapsing to distinct documents first would make recall@k meaningless here,
where the whole demo corpus is only four documents.
"""


def distinct_sources(chunk_sources: list[str]) -> list[str]:
    """Ranked list of distinct sources — for readable reports, not for scoring."""
    seen, out = set(), []
    for s in chunk_sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the relevant sources reached by the top k chunks.

    With a single relevant source this is 1.0 (a chunk of it is in the top k)
    or 0.0 (it never made it into the context).
    """
    if not relevant:
        raise ValueError("recall_at_k needs at least one relevant source")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    """1/position of the first chunk belonging to a relevant source, else 0.0.

    Rewards ranking the right document first rather than merely somewhere in
    the list — recall@k alone cannot tell those apart.
    """
    for position, source in enumerate(ranked, start=1):
        if source in relevant:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(results: list[tuple[str, list[str], list[str]]], ks: tuple[int, ...]) -> dict:
    """Score a whole question set.

    `results` is (question, relevant_sources, ranked_chunk_sources) per question.
    Returns aggregate metrics plus the questions that failed at the largest k,
    so a failing test can say *which* questions broke rather than just a number.
    """
    per_question, misses = [], []
    for question, relevant, ranked in results:
        row = {
            "question": question,
            "relevant": relevant,
            "ranked": distinct_sources(ranked),
            "rr": reciprocal_rank(ranked, relevant),
            **{f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks},
        }
        per_question.append(row)
        if row[f"recall@{max(ks)}"] == 0.0:
            misses.append(row)

    report = {f"recall@{k}": mean([r[f"recall@{k}"] for r in per_question]) for k in ks}
    report["mrr"] = mean([r["rr"] for r in per_question])
    report["n"] = len(per_question)
    report["per_question"] = per_question
    report["misses"] = misses
    return report


def format_report(report: dict, title: str) -> str:
    ks = sorted(int(key.split("@")[1]) for key in report if key.startswith("recall@"))
    lines = [f"\n{title}  (n={report['n']})"]
    lines += [f"  recall@{k}: {report[f'recall@{k}']:.3f}" for k in ks]
    lines.append(f"  MRR      : {report['mrr']:.3f}")
    if report["misses"]:
        lines.append(f"  missed ({len(report['misses'])}):")
        for m in report["misses"]:
            lines.append(f"    {m['question']!r}")
            lines.append(f"      expected {m['relevant']}, got {m['ranked'][:3]}")
    return "\n".join(lines)
