"""Retrieval quality on the demo corpus in docs/.

No API key and no network: embeddings run locally and the index is built from
docs/ into a temp directory, so this is reproducible in a fresh clone.

Thresholds are set from measured behaviour, with the margin sized to separate
the current setup from a known-bad one. Measured on the same 32 questions:

                      recall@1   recall@8    MRR
    multilingual        0.812      0.969    0.862     <- current
    bge-small-en        0.438      0.938    0.617     <- English-only, was shipped
    ... Russian only    0.190      0.905    0.440

Note how little recall@8 moves: with 24 chunks in the corpus, eight of them is a
third of everything, so the right document turns up almost by accident. recall@1
and MRR are the metrics that actually discriminate here — thresholds lean on them.
"""

import pytest

from ragbot.evaluation import evaluate, format_report

KS = (1, 4, 8)


def is_russian(text: str) -> bool:
    return any("а" <= c <= "я" for c in text.lower())


@pytest.fixture(scope="module")
def report(eval_set, ranked_sources):
    results = [(q["q"], q["expected"], ranked_sources(q["q"]))
               for q in eval_set["questions"]]
    return evaluate(results, ks=KS), results


def test_finds_the_right_document_first(report):
    full, _ = report
    assert full["recall@1"] >= 0.70, format_report(full, "recall@1 regressed")


def test_ranks_it_high_not_merely_somewhere(report):
    full, _ = report
    assert full["mrr"] >= 0.75, format_report(full, "MRR regressed")


def test_almost_nothing_falls_out_of_the_context_window(report):
    full, _ = report
    assert full["recall@8"] >= 0.90, format_report(full, "recall@8 regressed")


def test_russian_questions_are_not_second_class(report):
    """The guard that matters most.

    Russian questions against English documents is the real usage pattern, and
    it is what an English-only embedding model fails at while looking fine on
    the aggregate. These thresholds sit between the two measured configurations:
    the English-only model scored 0.190 / 0.440 here.
    """
    _, results = report
    russian = [r for r in results if is_russian(r[0])]
    assert len(russian) >= 15, "the Russian half of the eval set went missing"

    ru = evaluate(russian, ks=KS)
    assert ru["recall@1"] >= 0.55, format_report(ru, "Russian recall@1 regressed")
    assert ru["mrr"] >= 0.65, format_report(ru, "Russian MRR regressed")


def test_english_questions_stay_perfect(report):
    _, results = report
    english = evaluate([r for r in results if not is_russian(r[0])], ks=KS)
    assert english["recall@1"] >= 0.85, format_report(english, "English recall@1 regressed")


def test_every_document_is_reachable(report):
    """A document nothing can retrieve is invisible, however good the averages."""
    _, results = report
    reached = {s for _, _, ranked in results for s in ranked[:1]}
    expected = {s for _, relevant, _ in results for s in relevant}
    assert expected <= reached, f"never retrieved first: {sorted(expected - reached)}"


def test_out_of_domain_questions_land_much_further_away(eval_set, demo_store):
    """Nothing in docs/ answers these, and it should show in the distances.

    Compared as a ratio, not an absolute: the distance scale depends on whether
    the embedding model normalises its vectors, so absolute numbers are not
    portable across models. Measured ratio is 2.11 for the current model
    (in-domain mean 12.6 vs out-of-domain 26.6).

    Per-question ranges do overlap — the worst in-domain question is further
    away than the best out-of-domain one — so this is deliberately an assertion
    about the aggregate, not a usable per-question "is this answerable" cutoff.
    """
    def best_distance(q: str) -> float:
        return demo_store.similarity_search_with_score(q, k=1)[0][1]

    in_domain = [best_distance(q["q"]) for q in eval_set["questions"]]
    out_domain = [best_distance(q) for q in eval_set["negative"]]

    ratio = (sum(out_domain) / len(out_domain)) / (sum(in_domain) / len(in_domain))
    assert ratio >= 1.5, (
        f"out-of-domain questions are no longer clearly further away: "
        f"ratio {ratio:.2f} (in-domain mean {sum(in_domain)/len(in_domain):.2f}, "
        f"out-of-domain mean {sum(out_domain)/len(out_domain):.2f})"
    )
