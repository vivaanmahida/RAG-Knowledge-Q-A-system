import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.cache import (normalize, extract_numbers, has_negation, key_terms, is_safe_match,
                       build_entity_vocab, build_topic_vocab, guard_setup, kb_version,
                       QueryCache, answer_with_cache)
from src.classifier_checks import cross_validate
from src.db import get_connection, create_tables, save_chunk
from src.tune_cache_threshold import evaluate_threshold, choose_setting

VOCAB = {"margin", "padding", "alt", "filter", "map"}


VECTORS = {
    "what does margin do": [1.0, 0.0, 0.0],
    "what is the purpose of margin": [0.97, 0.10, 0.0],
    "what does padding do": [0.98, 0.08, 0.0],
    "how do i bake a cake": [0.0, 0.0, 1.0],
}


def fake_embed(texts):
    return [VECTORS.get(normalize(t), [0.0, 1.0, 0.0]) for t in texts]


def make_result(answer="Margin adds space [1].", refused=False):
    return {"answer": answer, "refused": refused, "reason": "ok", "best_score": 0.8,
            "scope_probability": None, "sources": [], "retrieved": [], "warning": None}


def new_cache(**kwargs):
    conn = get_connection(":memory:")
    return QueryCache(conn, embed_function=fake_embed, vocab=VOCAB, **kwargs)


def test_normalize():
    assert normalize("  What is   MARGIN?? ") == "what is margin"


def test_numbers_and_negation_helpers():
    assert extract_numbers("set 12 px and 3.5 em") == {"12", "3.5"}
    assert has_negation("a variable that cannot change")
    assert has_negation("it doesn't work")
    assert has_negation("it doesn’t work")
    assert not has_negation("a variable that can change")


def test_key_terms_only_keeps_rare_words():
    assert key_terms("what does the margin do", VOCAB) == {"margin"}


def test_safe_match_blocks_different_key_term_number_or_negation():
    assert is_safe_match("what does margin do", "what is the purpose of margin", VOCAB)
    assert not is_safe_match("what does margin do", "what does padding do", VOCAB)
    assert not is_safe_match("font size 12 pixels", "font size 20 pixels", VOCAB)
    assert not is_safe_match("variable that can change", "variable that cannot change", VOCAB)


def test_entity_vocab_keeps_rare_words_and_drops_common_ones():
    conn = get_connection(":memory:")
    create_tables(conn)
    for i in range(10):
        text = "the element is common" + (" margin" if i == 0 else "")
        save_chunk(conn, {"chunk_id": f"id{i}", "source_url": "u", "title": "t", "scraped_at": "d",
                          "chunk_size": 200, "chunk_index": i, "text": text})
    vocab = build_entity_vocab(conn, chunk_size=200, max_fraction=0.15)
    assert "margin" in vocab
    assert "common" not in vocab


def test_exact_hit_ignores_case_and_punctuation():
    cache = new_cache(threshold=0.9)
    cache.put("What does margin do?", make_result())
    result, kind, similarity = cache.get("what does MARGIN do")
    assert kind == "exact" and similarity == 1.0 and result["answer"].startswith("Margin")


def test_semantic_hit_for_paraphrase():
    cache = new_cache(threshold=0.9)
    cache.put("what does margin do", make_result())
    result, kind, similarity = cache.get("what is the purpose of margin")
    assert kind == "semantic" and similarity >= 0.9


def test_similar_looking_but_different_question_is_NOT_a_hit():
    cache = new_cache(threshold=0.9)
    cache.put("what does margin do", make_result())
    result, kind, similarity = cache.get("what does padding do")
    assert similarity > 0.9
    assert result is None


def test_without_guards_the_wrong_answer_would_be_reused():
    cache = new_cache(threshold=0.9, guards=())
    cache.put("what does margin do", make_result())
    assert cache.get("what does padding do")[1] == "semantic"


def test_below_threshold_is_a_miss():
    cache = new_cache(threshold=0.9)
    cache.put("what does margin do", make_result())
    assert cache.get("how do i bake a cake")[0] is None


def test_refusals_are_not_cached():
    cache = new_cache()
    assert cache.put("how do i bake a cake", make_result(refused=True)) is False
    assert cache.get("how do i bake a cake")[0] is None


def test_changed_knowledge_base_invalidates_old_answers():
    conn = get_connection(":memory:")
    old = QueryCache(conn, embed_function=fake_embed, vocab=VOCAB, version="v1")
    old.put("what does margin do", make_result())
    new = QueryCache(conn, embed_function=fake_embed, vocab=VOCAB, version="v2")
    assert new.get("what does margin do")[0] is None
    assert old.get("what does margin do")[0] is not None


def test_kb_version_changes_when_chunks_change():
    conn = get_connection(":memory:")
    create_tables(conn)
    chunk = {"chunk_id": "a", "source_url": "u", "title": "t", "scraped_at": "d",
             "chunk_size": 200, "chunk_index": 0, "text": "x"}
    save_chunk(conn, chunk)
    before = kb_version(conn, 200)
    save_chunk(conn, {**chunk, "chunk_id": "b"})
    assert kb_version(conn, 200) != before


def test_clear_empties_the_cache():
    cache = new_cache()
    cache.put("what does margin do", make_result())
    cache.clear()
    assert cache.get("what does margin do")[0] is None


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, json_mode=False):
        self.calls += 1
        return "Margin adds space outside an element [1]."


class OneChunkRetriever:
    def search(self, query, k=5):
        return [{"chunk_id": "c", "source_url": "http://x", "title": "T", "score": 0.9,
                 "text": "margin adds space"}]


def test_answer_with_cache_calls_the_llm_only_once():
    cache = new_cache(threshold=0.9)
    llm = CountingLLM()
    first = answer_with_cache("What does margin do?", OneChunkRetriever(), cache, llm=llm)
    second = answer_with_cache("what does margin do", OneChunkRetriever(), cache, llm=llm)
    third = answer_with_cache("what is the purpose of margin", OneChunkRetriever(), cache, llm=llm)
    other = answer_with_cache("what does padding do", OneChunkRetriever(), cache, llm=llm)

    assert llm.calls == 2
    assert first["cache"]["hit"] is False
    assert second["cache"] == {"hit": True, "kind": "exact", "similarity": 1.0}
    assert third["cache"]["kind"] == "semantic"
    assert other["cache"]["hit"] is False
    assert cache.stats == {"exact": 1, "semantic": 1, "miss": 2}


def add_chunks(conn, titles):
    create_tables(conn)
    for i, title in enumerate(titles):
        save_chunk(conn, {"chunk_id": f"id{i}", "source_url": "u", "title": title, "scraped_at": "d",
                          "chunk_size": 200, "chunk_index": i, "text": "some text"})


def test_topic_vocab_comes_from_titles_without_filler_words():
    conn = get_connection(":memory:")
    add_chunks(conn, ["margin", "<div>: The Content Division element",
                      "Array.prototype.map()", "Array.prototype.filter()"])
    vocab = build_topic_vocab(conn)
    assert {"margin", "div", "division", "map", "filter"} <= vocab
    assert "the" not in vocab and "element" not in vocab and "array" not in vocab


def test_guard_setup_modes():
    conn = get_connection(":memory:")
    add_chunks(conn, ["margin", "padding"])
    assert guard_setup(conn, "none") == (set(), ())
    assert guard_setup(conn, "numbers+negation")[1] == ("numbers", "negation")
    vocab, guards = guard_setup(conn, "topic")
    assert "margin" in vocab and "terms" in guards


def test_topic_guard_allows_paraphrase_but_blocks_other_topic():
    topic_vocab = {"margin", "padding", "map", "filter"}
    guards = ("numbers", "negation", "terms")

    assert is_safe_match("what does margin do", "what is the role and purpose of margin", topic_vocab, guards)
    assert not is_safe_match("what does margin do", "what does padding do", topic_vocab, guards)


def test_guards_can_be_switched_off_one_by_one():
    assert is_safe_match("set 12 px", "set 20 px", VOCAB, guards=())
    assert not is_safe_match("set 12 px", "set 20 px", VOCAB, guards=("numbers",))
    assert is_safe_match("a variable that can change", "a variable that cannot change", VOCAB, guards=("numbers",))


PAIRS = [
    {"question_a": "what does margin do", "question_b": "what is the purpose of margin", "label": 1},
    {"question_a": "what does margin do", "question_b": "what does padding do", "label": 0},
]


def test_evaluate_threshold_counts_right_and_wrong_reuse():
    sims = [0.95, 0.96]
    no_guard = evaluate_threshold(PAIRS, sims, 0.9, VOCAB, guards=(), mode="none")
    with_guard = evaluate_threshold(PAIRS, sims, 0.9, VOCAB, guards=("numbers", "negation", "terms"), mode="x")
    assert no_guard["reused_wrongly"] == 1 and no_guard["precision"] == 0.5
    assert with_guard["reused_wrongly"] == 0 and with_guard["precision"] == 1.0
    assert with_guard["recall"] == 1.0


def test_threshold_is_applied_before_guards():
    result = evaluate_threshold(PAIRS, [0.85, 0.85], 0.9, VOCAB, guards=(), mode="none")
    assert result["reused_correctly"] == 0 and result["reused_wrongly"] == 0


def test_choose_setting_prefers_safe_then_most_reuse():
    rows = [
        {"mode": "none", "threshold": 0.90, "reused_wrongly": 3, "recall": 0.9},
        {"mode": "rare", "threshold": 0.80, "reused_wrongly": 0, "recall": 0.03},
        {"mode": "topic", "threshold": 0.90, "reused_wrongly": 0, "recall": 0.60},
        {"mode": "topic", "threshold": 0.95, "reused_wrongly": 0, "recall": 0.60},
    ]
    assert choose_setting(rows) == ("topic", 0.90)


def test_choose_setting_when_nothing_is_safe_takes_fewest_wrong():
    rows = [
        {"mode": "none", "threshold": 0.90, "reused_wrongly": 5, "recall": 0.9},
        {"mode": "none", "threshold": 0.95, "reused_wrongly": 1, "recall": 0.5},
    ]
    assert choose_setting(rows) == ("none", 0.95)


def test_cross_validate_gives_one_score_per_fold():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(1, 0.1, (20, 3)), rng.normal(-1, 0.1, (20, 3))])
    y = np.array([1] * 20 + [0] * 20)
    scores = cross_validate(X, y, C=1.0, folds=5)
    assert len(scores) == 5 and scores.min() == 1.0
