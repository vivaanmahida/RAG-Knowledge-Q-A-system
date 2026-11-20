


import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from src.classifier import ScopeClassifier, load_data, embed_questions, evaluate


def cross_validate(X, y, C=1.0, folds=5):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
    model = LogisticRegression(C=C, max_iter=1000)
    return cross_val_score(model, X, y, cv=splitter, scoring="f1")


def test_on_hard_set(classifier, questions, labels):
    probabilities = np.array([classifier.in_scope_probability(q) for q in questions])
    metrics = evaluate(labels, probabilities, classifier.threshold)
    mistakes = []
    for q, p, label in zip(questions, probabilities, labels):
        if int(p >= classifier.threshold) != label:
            mistakes.append({"question": q, "true_label": int(label), "probability_in_scope": round(float(p), 3)})
    return metrics, mistakes


def main():
    results = {}

    questions, labels = load_data(config.CLASSIFIER_DATA_PATH)
    X = embed_questions(questions)
    with open(config.CLASSIFIER_RESULTS_PATH) as f:
        chosen_c = json.load(f)["chosen_C"]
    scores = cross_validate(X, labels, C=chosen_c)
    results["cross_validation_f1"] = {
        "per_fold": [round(float(s), 3) for s in scores],
        "mean": round(float(scores.mean()), 3),
        "std": round(float(scores.std()), 3),
        "C": chosen_c,
    }
    print("5-fold cross-validation F1:", results["cross_validation_f1"])

    if os.path.exists(config.HARD_TEST_PATH):
        hard_questions, hard_labels = load_data(config.HARD_TEST_PATH)
        metrics, mistakes = test_on_hard_set(ScopeClassifier(), hard_questions, hard_labels)
        results["hard_test"] = {"size": len(hard_questions), **metrics, "mistakes": mistakes}
        print(f"\nHard test ({len(hard_questions)} questions):", metrics)
        for m in mistakes:
            print("  mistake:", m)
    else:
        print(f"\n{config.HARD_TEST_PATH} not found. Write about 20 questions (question,label) and run again.")

    os.makedirs("results", exist_ok=True)
    with open(config.CLASSIFIER_CHECKS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
