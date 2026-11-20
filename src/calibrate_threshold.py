


import csv

import numpy as np

import config
from src.retrieval import DenseRetriever


OUT_OF_SCOPE = [
    "Who won the cricket world cup in 2011?",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "What is the price of Bitcoin today?",
    "Who is the prime minister of India?",
    "Explain the theory of relativity.",
    "How many calories are in a banana?",
    "What is the best movie of all time?",
    "How do I train for a marathon?",
    "Translate hello into Spanish.",
    "How do I use CSS grid to make a layout?",
    "What is a React component?",
    "How do I connect Python to a MySQL database?",
    "What does the HTML video tag do?",
    "How do I center a div with Tailwind CSS?",
    "What is a JavaScript promise?",
    "How do I deploy a website to Netlify?",
    "What is the difference between SQL and NoSQL?",
    "How do I use git rebase?",
    "What is Node.js used for?",
]


def best_scores(retriever, questions):
    return [retriever.search(q, 1)[0]["score"] for q in questions]


def main():
    with open(config.EVAL_PATH, newline="", encoding="utf-8") as f:
        in_scope = [row["question"] for row in csv.DictReader(f)]

    retriever = DenseRetriever(config.RAG_CHUNK_SIZE)
    in_scores = np.array(best_scores(retriever, in_scope))
    out_scores = np.array(best_scores(retriever, OUT_OF_SCOPE))

    print("OUT-OF-SCOPE questions (best similarity each):")
    for q, s in sorted(zip(OUT_OF_SCOPE, out_scores), key=lambda x: -x[1]):
        print(f"  {s:.3f}  {q}")

    print(f"\nIn-scope  : min {in_scores.min():.3f} | 5th percentile {np.percentile(in_scores, 5):.3f} | median {np.median(in_scores):.3f}")
    print(f"Out-scope : max {out_scores.max():.3f} | median {np.median(out_scores):.3f}")

    low_in = np.percentile(in_scores, 5)
    suggestion = (low_in + out_scores.max()) / 2
    print(f"\nSuggested REFUSE_THRESHOLD = {suggestion:.2f}")
    if out_scores.max() >= low_in:
        print("Note: the two groups overlap, so a score threshold alone cannot separate")
        print("them. The classifier (next step) is there to fix exactly this.")


if __name__ == "__main__":
    main()
