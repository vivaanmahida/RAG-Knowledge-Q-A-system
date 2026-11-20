


import math


def normalize(text):
    return " ".join(text.lower().split())


def is_relevant(chunk_text, evidence):
    return normalize(evidence) in normalize(chunk_text)


def first_hit_only(relevance):
    result = []
    found = False
    for r in relevance:
        if r and not found:
            result.append(1)
            found = True
        else:
            result.append(0)
    return result


def recall_at_k(relevance, k):
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance):
    for position, r in enumerate(relevance):
        if r:
            return 1.0 / (position + 1)
    return 0.0


def ndcg_at_k(relevance, k):
    relevance = first_hit_only(relevance[:k])
    dcg = 0.0
    for position, r in enumerate(relevance):
        dcg += r / math.log2(position + 2)
    return dcg


def average(values):
    return sum(values) / len(values) if values else 0.0
