#!/usr/bin/env python3
"""CPU-only synthetic data audit. No download, model calls, or code execution.

Run from any directory: python3 examples/data_audit.py
The output is a teaching fixture, NOT an audit of OMR/OCR/TextbookReasoning.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import unicodedata


def normalize(text):
    # Do not lowercase code, remove signs, or discard mathematical punctuation.
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def exact_numeric_equal(prediction, reference):
    """A deliberately narrow verifier for rational-number strings only."""
    try:
        return Fraction(prediction.strip()) == Fraction(reference.strip())
    except (ValueError, ZeroDivisionError):
        return None  # Unsupported != wrong. Do not eval untrusted strings.


def row(rid, family, question, answer, expected, process, verdict, calls):
    return dict(id=rid, family_id=family, question=question, answer=answer,
                expected_answer=expected, process_status=process,
                checker_status=verdict, calls=calls)


def fixture():
    # Families are known because we authored the synthetic data.
    # In a real dataset, these require source IDs or human/semantic investigation.
    return [
        row("a1", "add-2-3", "What is 2 + 3?", "5", "5", "unchecked", "pass", [(8, 3)]),
        row("a2", "add-2-3", "What  is 2 + 3?", "6", "5", "invalid", "fail", [(8, 9)]),
        row("a3", "add-2-3", "求二与三的和。", "5", "5", "valid", "pass", [(8, 6)]),
        row("b1", "divide-1-2", "What is 1 / 2?", "0.5", "1/2", "valid", "pass", [(9, 5)]),
        row("b2", "divide-1-2", "Compute half of one.", "1/2", "1/2", "unchecked", "pass", [(9, 20)]),
        row("c1", "add-4-3", "What is 4 + 3?", "7", "7", "unchecked", "pass", [(8, 3)]),
        row("c2", "add-4-3", "求四与三的和。", "7", "7", "invalid", "pass", [(8, 18)]),
        row("d1", "proof-even", "Prove the sum of two even integers is even.",
            "2a + 2b = 2(a+b)", "proof", "unchecked", "unsupported", [(15, 21)]),
    ]


def audit(rows, heldout):
    by_question = defaultdict(list)
    for item in rows:
        by_question[digest(item["question"])].append(item)
    duplicate_groups = [group for group in by_question.values() if len(group) > 1]
    conflicts = [group for group in duplicate_groups
                 if len({normalize(x["answer"]) for x in group}) > 1]
    heldout_hashes = {digest(x["question"]) for x in heldout}
    heldout_families = {x["family_id"] for x in heldout}
    exact_hits = [x["id"] for x in rows if digest(x["question"]) in heldout_hashes]
    family_hits = [x["id"] for x in rows if x["family_id"] in heldout_families]
    candidates = [x for x in rows if x["family_id"] not in heldout_families]
    return {
        "rows": len(rows),
        "known_families": len({x["family_id"] for x in rows}),
        "duplicate_question_groups": [[x["id"] for x in g] for g in duplicate_groups],
        "conflicting_answer_groups": [[x["id"] for x in g] for g in conflicts],
        "exact_test_overlap_ids": exact_hits,
        "known_family_test_overlap_ids": family_hits,
        "candidate_ids_after_family_exclusion": [x["id"] for x in candidates],
        "process_status": dict(Counter(x["process_status"] for x in rows)),
        "checker_status": dict(Counter(x["checker_status"] for x in rows)),
    }


def main():
    print("SYNTHETIC TEACHING FIXTURE — not a real corpus audit")
    rows = fixture()
    heldout = [{"question": "What is 2 + 3?", "family_id": "add-2-3"}]
    report = audit(rows, heldout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["exact_test_overlap_ids"] == ["a1", "a2"]
    assert report["known_family_test_overlap_ids"] == ["a1", "a2", "a3"]
    assert exact_numeric_equal("0.5", "1/2") is True
    assert exact_numeric_equal("2a + 2b", "proof") is None
    assert report["conflicting_answer_groups"] == [["a1", "a2"]]

    print("\nSCORING DEMO — label and verifier errors have different directions")
    # This is a fabricated confusion matrix; the 'truth' is fixture metadata.
    truth = [True, True, False, False]
    judge = [True, False, True, False]
    fp = sum((not t) and j for t, j in zip(truth, judge))
    fn = sum(t and (not j) for t, j in zip(truth, judge))
    print(f"true_success={sum(truth) / len(truth):.2f}, "
          f"judge_success={sum(judge) / len(judge):.2f}, FP={fp}, FN={fn}")
    print("Equal aggregate scores can hide different misjudged examples.")

    print("\nTOTAL REQUEST TOKEN ACCOUNTING — fabricated call logs")
    strategies = {
        "one_call": [[(100, 60)], [(100, 80)]],
        "shorter_final_with_retry": [[(100, 20), (140, 25)], [(100, 20), (140, 25)]],
    }
    for name, requests in strategies.items():
        costs = [sum(inp + out for inp, out in calls) for calls in requests]
        finals = [calls[-1][1] for calls in requests]
        print(f"{name}: mean_total={sum(costs) / len(costs):.1f}, "
              f"mean_final_output={sum(finals) / len(finals):.1f}")
    print("No performance claim: we have not run a language model.")


if __name__ == "__main__":
    main()
