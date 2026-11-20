import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.refusal_checks import run_checks, summarize


class ScoreRetriever:
    def __init__(self, scores):
        self.scores = scores

    def search(self, query, k=5):
        return [{"chunk_id": "c", "source_url": "u", "title": "T", "text": "text",
                 "score": self.scores[query]}]


class FakeClassifier:
    def __init__(self, in_scope_questions):
        self.in_scope_questions = in_scope_questions

    def is_in_scope(self, question):
        ok = question in self.in_scope_questions
        return ok, 0.9 if ok else 0.1


def fake_llm(prompt, json_mode=False):
    if "needs refusal" in prompt:
        return "I don't have enough information in the knowledge base to answer that."
    return "Some answer [1]."


def test_each_layer_is_counted_and_summary_is_correct():
    questions = ["weather today",
                 "css opacity",
                 "needs refusal span",
                 "css shadow",
                 "how to make text bigger",
                 "let vs const"]
    labels = [0, 0, 0, 0, 1, 1]
    scores = {"weather today": 0.2, "css opacity": 0.8, "needs refusal span": 0.8,
              "css shadow": 0.8, "how to make text bigger": 0.8, "let vs const": 0.8}
    classifier = FakeClassifier({"needs refusal span", "css shadow", "how to make text bigger"})

    rows = run_checks(questions, labels, ScoreRetriever(scores), classifier, llm=fake_llm)
    summary = summarize(rows)

    assert summary["out_of_scope_questions"] == 4
    assert summary["out_of_scope_stopped_by"] == {
        "layer 1: similarity threshold": 1,
        "layer 2: classifier": 1,
        "layer 3: LLM refusal": 1,
    }
    assert summary["out_of_scope_wrongly_answered"] == 1
    assert summary["in_scope_questions"] == 2
    assert summary["in_scope_wrongly_refused"] == 1
    assert summary["in_scope_refused_by"] == {"layer 2: classifier": 1}
