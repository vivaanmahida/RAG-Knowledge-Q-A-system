


import csv
import os

import config
from src.metrics import is_relevant, recall_at_k, reciprocal_rank, ndcg_at_k, average
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def load_eval_set(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(retriever, eval_set, k):
    recalls, rrs, ndcgs = [], [], []

    for row in eval_set:
        results = retriever.search(row["question"], k)
        relevance = [1 if is_relevant(r["text"], row["evidence"]) else 0 for r in results]

        recalls.append(recall_at_k(relevance, k))
        rrs.append(reciprocal_rank(relevance))
        ndcgs.append(ndcg_at_k(relevance, k))

    return average(recalls), average(rrs), average(ndcgs)


def main():
    eval_set = load_eval_set(config.EVAL_PATH)
    k = config.TOP_K
    print(f"{len(eval_set)} questions, k = {k}\n")

    table = []
    number = 1
    for size in config.CHUNK_SIZES:
        for name, retriever_class in [("BM25", BM25Retriever), ("Dense", DenseRetriever),
                                      ("Hybrid", HybridRetriever)]:
            retriever = retriever_class(size)
            recall, mrr, ndcg = evaluate(retriever, eval_set, k)
            table.append({
                "Configuration": f"Experiment {number}",
                "Chunk Size": size,
                "Retriever": name,
                f"Recall@{k}": round(recall, 3),
                "MRR": round(mrr, 3),
                f"nDCG@{k}": round(ndcg, 3),
            })
            print(table[-1])
            number += 1

    os.makedirs("results", exist_ok=True)
    with open(config.RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print("\nSaved to", config.RESULTS_PATH)


if __name__ == "__main__":
    main()
