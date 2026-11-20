import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag import (answer_question, build_context, find_cited_numbers,
                     llm_said_no, NO_ANSWER)


class FakeRetriever:
    def __init__(self, scores):
        self.chunks = [
            {"chunk_id": f"id{i}", "source_url": f"http://x/{i}", "title": f"Page{i}",
             "text": f"text number {i}", "score": s}
            for i, s in enumerate(scores)
        ]

    def search(self, query, k=5):
        return self.chunks[:k]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.called = False

    def __call__(self, prompt):
        self.called = True
        return self.reply


def test_low_similarity_refuses_and_does_not_call_llm():
    llm = FakeLLM("should not be used")
    result = answer_question("who won the world cup?", FakeRetriever([0.2, 0.1]), llm, threshold=0.5)
    assert result["refused"] is True
    assert result["reason"] == "low_similarity"
    assert result["answer"] == NO_ANSWER
    assert llm.called is False


def test_good_answer_returns_cited_sources_only():
    llm = FakeLLM("The margin adds space outside the border [2].")
    result = answer_question("what is margin?", FakeRetriever([0.9, 0.8, 0.7]), llm, threshold=0.5)
    assert result["refused"] is False
    assert [s["chunk_id"] for s in result["sources"]] == ["id1"]
    assert len(result["retrieved"]) == 3


def test_llm_refusal_is_detected():
    llm = FakeLLM("I don’t have enough information in the knowledge base to answer that.")
    result = answer_question("something odd", FakeRetriever([0.9]), llm, threshold=0.5)
    assert result["refused"] is True
    assert result["reason"] == "llm_refused"


def test_answer_without_citation_gets_warning():
    llm = FakeLLM("Margin adds space.")
    result = answer_question("what is margin?", FakeRetriever([0.9]), llm, threshold=0.5)
    assert result["refused"] is False
    assert result["warning"] is not None
    assert result["sources"] == []


def test_empty_question_is_refused():
    result = answer_question("   ", FakeRetriever([0.9]), FakeLLM("x"))
    assert result["refused"] is True
    assert result["reason"] == "empty_question"


def test_find_cited_numbers_ignores_wrong_numbers():
    assert find_cited_numbers("fact [1] and fact [3] and fake [9]", 3) == [1, 3]


def test_build_context_is_numbered():
    chunks = FakeRetriever([0.9, 0.8]).chunks
    context = build_context(chunks)
    assert "[1] (Page0)" in context and "[2] (Page1)" in context


def test_llm_said_no_handles_both_apostrophes():
    assert llm_said_no("I don't have enough information")
    assert llm_said_no("I don’t have enough information")
    assert not llm_said_no("Margin adds space [1].")


class FakeClassifier:
    def __init__(self, in_scope, probability):
        self.in_scope, self.probability = in_scope, probability

    def is_in_scope(self, question):
        return self.in_scope, self.probability


def test_classifier_refuses_near_topic_question_even_if_similarity_is_high():
    llm = FakeLLM("should not be used")
    result = answer_question("how do I use css grid?", FakeRetriever([0.62]), llm,
                             threshold=0.5, classifier=FakeClassifier(False, 0.12))
    assert result["refused"] is True
    assert result["reason"] == "out_of_scope_classifier"
    assert result["scope_probability"] == 0.12
    assert llm.called is False


def test_classifier_allows_in_scope_question():
    llm = FakeLLM("Margin adds space [1].")
    result = answer_question("what is margin?", FakeRetriever([0.8]), llm,
                             threshold=0.5, classifier=FakeClassifier(True, 0.93))
    assert result["refused"] is False
    assert result["scope_probability"] == 0.93
