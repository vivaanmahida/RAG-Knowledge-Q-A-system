


import csv
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import config
from src.embedding import create_embedding


def load_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    questions = [r["question"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    return questions, labels


def embed_questions(questions, embed_function=create_embedding):
    vectors = []
    for start in range(0, len(questions), config.EMBED_BATCH_SIZE):
        vectors.extend(embed_function(questions[start:start + config.EMBED_BATCH_SIZE]))
    return np.array(vectors)


def split_data(labels, seed=config.RANDOM_SEED):
    indexes = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        indexes, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=labels[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def train_model(X, y, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X, y)
    return model


def evaluate(y_true, probabilities, threshold=config.CLASSIFIER_THRESHOLD):
    predicted = (np.array(probabilities) >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 3),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def run_training(questions, labels, embed_function=create_embedding, baseline_scorer=None):
    X = embed_questions(questions, embed_function)
    train_idx, val_idx, test_idx = split_data(labels)


    best_C, best_f1, best_model = None, -1, None
    tried = {}
    for C in [0.1, 1, 10]:
        model = train_model(X[train_idx], labels[train_idx], C)
        probs = model.predict_proba(X[val_idx])[:, 1]
        val_f1 = evaluate(labels[val_idx], probs)["f1"]
        tried[str(C)] = val_f1
        if val_f1 > best_f1:
            best_C, best_f1, best_model = C, val_f1, model


    test_probs = best_model.predict_proba(X[test_idx])[:, 1]
    results = {
        "model": "LogisticRegression on bge-m3 embeddings",
        "sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "validation_f1_for_each_C": tried,
        "chosen_C": best_C,
        "threshold": config.CLASSIFIER_THRESHOLD,
        "test": evaluate(labels[test_idx], test_probs),
    }


    wrong = []
    for i, p in zip(test_idx, test_probs):
        if int(p >= config.CLASSIFIER_THRESHOLD) != labels[i]:
            wrong.append({"question": questions[i], "true_label": int(labels[i]),
                          "probability_in_scope": round(float(p), 3)})
    results["test_mistakes"] = wrong


    if baseline_scorer is not None:
        scores = np.array([baseline_scorer(questions[i]) for i in test_idx])
        results["baseline_similarity_rule"] = evaluate(
            labels[test_idx], (scores >= config.REFUSE_THRESHOLD).astype(float), threshold=0.5)

    return best_model, results


class ScopeClassifier:
    def __init__(self, model=None, threshold=None, path=config.CLASSIFIER_PATH,
                 embed_function=create_embedding):
        if model is None:
            saved = joblib.load(path)
            model, threshold = saved["model"], saved["threshold"]
        self.model = model
        self.threshold = threshold if threshold is not None else config.CLASSIFIER_THRESHOLD
        self.embed_function = embed_function

    def in_scope_probability(self, question):
        vector = np.array(self.embed_function([question]))
        return float(self.model.predict_proba(vector)[0][1])

    def is_in_scope(self, question):
        probability = self.in_scope_probability(question)
        return probability >= self.threshold, probability


def main():
    from src.retrieval import DenseRetriever

    questions, labels = load_data(config.CLASSIFIER_DATA_PATH)
    print(f"{len(questions)} questions: {int(labels.sum())} in-scope, {int((labels == 0).sum())} out-of-scope")

    retriever = DenseRetriever(config.RAG_CHUNK_SIZE)
    baseline = lambda q: retriever.search(q, 1)[0]["score"]

    model, results = run_training(questions, labels, baseline_scorer=baseline)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump({"model": model, "threshold": config.CLASSIFIER_THRESHOLD}, config.CLASSIFIER_PATH)
    with open(config.CLASSIFIER_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "test_mistakes"}, indent=2))
    print(f"\n{len(results['test_mistakes'])} mistakes on the test set:")
    for m in results["test_mistakes"]:
        print("  ", m)
    print("\nSaved model to", config.CLASSIFIER_PATH)


if __name__ == "__main__":
    main()
