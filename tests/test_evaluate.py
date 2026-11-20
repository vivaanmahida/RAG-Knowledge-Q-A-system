import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.compare_judge import agreement_rate, kappa, summarize
from src.evaluate_answers import (split_statements, judge_answer, count_unsupported,
                                  citation_correct, compute_metrics, badness,
                                  worst_answers, evaluate_one, to_bool)


def fake_judge(reply):
    return lambda prompt, json_mode=False: reply


def make_row(**changes):
    row = {"question": "q", "expected_answer": "a", "evidence": "e", "answer": "x", "refused": False,
           "reason": "ok", "judge_faithful": True, "judge_relevant": True,
           "unsupported_statements": 0, "total_statements": 2,
           "has_citation": True, "citation_correct": True, "evidence_in_retrieved": True,
           "retrieved_context": "ctx"}
    row.update(changes)
    return row


def test_to_bool():
    assert to_bool("Yes") is True and to_bool("no") is False and to_bool("maybe") is None


def test_split_statements_removes_markers_and_short_pieces():
    answer = "The margin adds space outside the border [1]. Yes. It never changes the size of the content [2]."
    assert split_statements(answer) == ["The margin adds space outside the border.",
                                        "It never changes the size of the content."]


def test_judge_answer_reads_json():
    assert judge_answer("q", "a", "ctx", fake_judge('{"faithful":"yes","relevant":"no"}')) == (True, False)


def test_judge_answer_survives_bad_reply():
    assert judge_answer("q", "a", "ctx", fake_judge("not json")) == (None, None)


def test_count_unsupported():
    answer = "The first sentence has enough words. The second sentence also has enough words."
    assert count_unsupported(answer, "ctx", fake_judge('{"supported":"no"}')) == (2, 2)
    assert count_unsupported(answer, "ctx", fake_judge('{"supported":"yes"}')) == (0, 2)
    assert count_unsupported(answer, "ctx", fake_judge("garbage")) == (0, 0)


def test_citation_correct():
    result = {"sources": [{"text": "the margin property adds space"}]}
    assert citation_correct(result, "margin property adds space") is True
    assert citation_correct(result, "something else entirely") is False
    assert citation_correct({"sources": []}, "anything") is False


def test_compute_metrics_numerators_and_denominators():
    rows = [
        make_row(),
        make_row(judge_faithful=False, unsupported_statements=1),
        make_row(citation_correct=False, has_citation=True),
        make_row(refused=True, judge_faithful=None, judge_relevant=None,
                 has_citation=None, citation_correct=None, total_statements=0),
    ]
    m = compute_metrics(rows)
    assert m["false_refusal_rate"] == {"value": 0.25, "numerator": 1, "denominator": 4}
    assert m["faithfulness"] == {"value": 0.667, "numerator": 2, "denominator": 3}
    assert m["hallucination_rate"] == {"value": 0.167, "numerator": 1, "denominator": 6}
    assert m["citation_correctness"] == {"value": 0.667, "numerator": 2, "denominator": 3}


def test_worst_answers_are_ranked_by_badness():
    good = make_row(question="good")
    bad = make_row(question="bad", judge_faithful=False, judge_relevant=False, citation_correct=False)
    refused = make_row(question="refused", refused=True)
    ranked = worst_answers([good, refused, bad], n=2)
    assert [r["question"] for r in ranked] == ["bad", "refused"]
    assert badness(good) == 0


def test_agreement_and_kappa():
    assert agreement_rate([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75
    assert kappa([1, 1, 1], [1, 1, 1]) is None
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_summarize_shows_judge_leniency():
    pairs = [{"human": 0, "judge": 1}, {"human": 1, "judge": 1}, {"human": 0, "judge": 1}, {"human": 0, "judge": 0}]
    s = summarize(pairs)
    assert s["agreement"] == 0.5
    assert s["human_yes_rate"] == 0.25 and s["judge_yes_rate"] == 0.75
    assert s["confusion_human_vs_judge"] == [[1, 2], [0, 1]]


class FakeRetriever:
    def search(self, query, k=5):
        return [{"chunk_id": "c1", "source_url": "http://x", "title": "T", "score": 0.9,
                 "text": "the margin property adds space outside the border of an element"}]


def test_evaluate_one_full_row_with_fake_models():
    item = {"question": "what does margin do?", "expected_answer": "adds space",
            "evidence": "margin property adds space"}
    answer_llm = lambda prompt, json_mode=False: "The margin property adds space outside the border [1]."
    judge_llm = fake_judge('{"faithful":"yes","relevant":"yes","supported":"yes"}')
    row = evaluate_one(item, FakeRetriever(), None, answer_llm, judge_llm)
    assert row["refused"] is False
    assert row["judge_faithful"] is True and row["judge_relevant"] is True
    assert row["citation_correct"] is True and row["has_citation"] is True
    assert row["evidence_in_retrieved"] is True
    assert row["total_statements"] == 1 and row["unsupported_statements"] == 0
