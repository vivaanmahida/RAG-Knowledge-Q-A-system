


import csv
import json
import os
from collections import Counter

import config
from src.classifier import ScopeClassifier, load_data
from src.llm import generate
from src.rag import answer_question
from src.retrieval import DenseRetriever

LAYER_NAMES = {
    "low_similarity": "layer 1: similarity threshold",
    "out_of_scope_classifier": "layer 2: classifier",
    "llm_refused": "layer 3: LLM refusal",
    "ok": "answered",
}


def run_checks(questions, labels, retriever, classifier, llm=generate):
    rows = []
    for question, label in zip(questions, labels):
        result = answer_question(question, retriever, llm=llm, classifier=classifier)
        rows.append({
            "question": question,
            "label": int(label),
            "outcome": LAYER_NAMES.get(result["reason"], result["reason"]),
            "best_score": round(result["best_score"], 3),
            "scope_probability": None if result["scope_probability"] is None
                                 else round(result["scope_probability"], 3),
            "answer": result["answer"],
        })
    return rows


def summarize(rows):
    out_of_scope = [r for r in rows if r["label"] == 0]
    in_scope = [r for r in rows if r["label"] == 1]
    answered = LAYER_NAMES["ok"]
    return {
        "out_of_scope_questions": len(out_of_scope),
        "out_of_scope_stopped_by": dict(Counter(r["outcome"] for r in out_of_scope if r["outcome"] != answered)),
        "out_of_scope_wrongly_answered": sum(1 for r in out_of_scope if r["outcome"] == answered),
        "in_scope_questions": len(in_scope),
        "in_scope_wrongly_refused": sum(1 for r in in_scope if r["outcome"] != answered),
        "in_scope_refused_by": dict(Counter(r["outcome"] for r in in_scope if r["outcome"] != answered)),
    }


def main():
    questions, labels = load_data(config.HARD_TEST_PATH)
    retriever = DenseRetriever(config.RAG_CHUNK_SIZE)
    classifier = ScopeClassifier() if os.path.exists(config.CLASSIFIER_PATH) else None

    rows = run_checks(questions, labels, retriever, classifier)
    summary = summarize(rows)

    os.makedirs("results", exist_ok=True)
    with open("results/refusal_checks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open("results/refusal_checks.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nOut-of-scope questions that were ANSWERED (read these answers yourself):")
    for r in rows:
        if r["label"] == 0 and r["outcome"] == LAYER_NAMES["ok"]:
            print(f"\n  Q: {r['question']}\n  A: {r['answer']}")
    print("\nIn-scope questions that were REFUSED:")
    for r in rows:
        if r["label"] == 1 and r["outcome"] != LAYER_NAMES["ok"]:
            print(f"  {r['question']}  ->  {r['outcome']}")
    print("\nDetails saved to results/refusal_checks.csv and results/refusal_checks.json")


if __name__ == "__main__":
    main()
