import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.classifier import (load_data, split_data, train_model, evaluate,
                            run_training, ScopeClassifier)


def fake_embed(texts):
    vectors = []
    for t in texts:
        in_scope = "css" in t.lower() or "html" in t.lower()
        base = np.array([1.0, 1.0, 0.0]) if in_scope else np.array([-1.0, -1.0, 0.0])
        vectors.append(base + 0.05 * (len(t) % 5))
    return vectors


def make_questions():
    questions = [f"what does css property number {i} do" for i in range(40)]
    questions += [f"who won football match number {i}" for i in range(40)]
    labels = np.array([1] * 40 + [0] * 40)
    return questions, labels


def test_load_data(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("question,label,source\nwhat is css?,1,x\nwho is the pm?,0,y\n")
    questions, labels = load_data(str(path))
    assert questions == ["what is css?", "who is the pm?"]
    assert list(labels) == [1, 0]


def test_split_is_70_15_15_without_overlap():
    _, labels = make_questions()
    train, val, test = split_data(labels)
    assert (len(train), len(val), len(test)) == (56, 12, 12)
    assert len(set(train) | set(val) | set(test)) == 80
    for part in (train, val, test):
        assert abs(labels[part].mean() - 0.5) < 0.1


def test_split_is_reproducible():
    _, labels = make_questions()
    assert list(split_data(labels)[2]) == list(split_data(labels)[2])


def test_evaluate_known_case():
    result = evaluate([1, 1, 0, 0], [0.9, 0.4, 0.2, 0.6], threshold=0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_model_learns_separable_data():
    X = np.array([[1.0, 1.0]] * 20 + [[-1.0, -1.0]] * 20)
    y = np.array([1] * 20 + [0] * 20)
    model = train_model(X, y)
    result = evaluate(y, model.predict_proba(X)[:, 1])
    assert result["f1"] == 1.0


def test_run_training_end_to_end_with_baseline():
    questions, labels = make_questions()
    model, results = run_training(questions, labels, embed_function=fake_embed,
                                  baseline_scorer=lambda q: 0.9 if "css" in q else 0.1)
    assert results["sizes"] == {"train": 56, "validation": 12, "test": 12}
    assert results["test"]["f1"] == 1.0
    assert "baseline_similarity_rule" in results
    assert results["test_mistakes"] == []
    json.dumps(results)


def test_scope_classifier_predicts_with_fake_embedder():
    questions, labels = make_questions()
    model, _ = run_training(questions, labels, embed_function=fake_embed)
    classifier = ScopeClassifier(model=model, threshold=0.5, embed_function=fake_embed)
    assert classifier.is_in_scope("how does css margin work")[0] is True
    assert classifier.is_in_scope("who won the football match")[0] is False
