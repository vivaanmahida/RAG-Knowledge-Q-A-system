


import csv
import os

import numpy as np

import config
from src.cache import cosine, is_safe_match, guard_setup, GUARD_MODES
from src.classifier import embed_questions
from src.db import get_connection

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.97]


TRAP_PAIRS = [
    ("How do I declare a variable that can be reassigned?", "How do I declare a variable that cannot be reassigned?"),
    ("What does the map method return?", "What does the filter method return?"),
    ("How do I add space outside an element?", "How do I add space inside an element?"),
    ("Which element creates a button?", "Which element creates a form?"),
    ("How do I set the font size to 12 pixels?", "How do I set the font size to 20 pixels?"),
    ("What does the const keyword do?", "What does the let keyword do?"),
]


def is_reused(pair, similarity, threshold, vocab, guards):
    if similarity < threshold:
        return False
    return is_safe_match(pair["question_a"], pair["question_b"], vocab, guards) if guards else True


def evaluate_threshold(pairs, similarities, threshold, vocab, guards, mode="custom"):
    true_hits = false_hits = positives = negatives = 0
    for pair, sim in zip(pairs, similarities):
        hit = is_reused(pair, sim, threshold, vocab, guards)
        if pair["label"] == 1:
            positives += 1
            true_hits += hit
        else:
            negatives += 1
            false_hits += hit
    total_hits = true_hits + false_hits
    return {
        "mode": mode, "threshold": threshold,
        "reused_correctly": int(true_hits), "same_meaning_pairs": positives,
        "reused_wrongly": int(false_hits), "different_pairs": negatives,
        "precision": round(true_hits / total_hits, 3) if total_hits else None,
        "recall": round(true_hits / positives, 3) if positives else None,
    }


def choose_setting(rows):
    best = max(rows, key=lambda r: (r["reused_wrongly"] == 0, -r["reused_wrongly"],
                                    r["recall"] or 0, -r["threshold"]))
    return best["mode"], best["threshold"]


def failures(pairs, similarities, threshold, vocab, guards):
    missed, wrong = [], []
    for pair, sim in zip(pairs, similarities):
        hit = is_reused(pair, sim, threshold, vocab, guards)
        if pair["label"] == 1 and not hit:
            missed.append((round(sim, 3), pair["question_a"], pair["question_b"]))
        if pair["label"] == 0 and hit:
            wrong.append((round(sim, 3), pair["question_a"], pair["question_b"]))
    return missed, wrong


def main():
    with open(config.CACHE_PAIRS_PATH, newline="", encoding="utf-8") as f:
        pairs = [{"question_a": r["question_a"], "question_b": r["question_b"], "label": int(r["label"])}
                 for r in csv.DictReader(f)]
    pairs += [{"question_a": a, "question_b": b, "label": 0} for a, b in TRAP_PAIRS]

    vectors_a = embed_questions([p["question_a"] for p in pairs])
    vectors_b = embed_questions([p["question_b"] for p in pairs])
    similarities = [cosine(a, b) for a, b in zip(vectors_a, vectors_b)]

    labels = np.array([p["label"] for p in pairs])
    sims = np.array(similarities)
    print(f"{len(pairs)} pairs ({int(labels.sum())} same meaning, {int((labels == 0).sum())} different)")
    print(f"similarity of same-meaning pairs : median {np.median(sims[labels == 1]):.3f}, min {sims[labels == 1].min():.3f}")
    print(f"similarity of different pairs    : median {np.median(sims[labels == 0]):.3f}, max {sims[labels == 0].max():.3f}\n")

    conn = get_connection()
    setups = {mode: guard_setup(conn, mode) for mode in GUARD_MODES}
    rows = []
    for mode in GUARD_MODES:
        vocab, guards = setups[mode]
        for t in THRESHOLDS:
            rows.append(evaluate_threshold(pairs, similarities, t, vocab, guards, mode))

    print(f"{'mode':18} {'T':>5} {'reused ok':>10} {'reused WRONG':>13} {'precision':>10} {'recall':>7}")
    for r in rows:
        print(f"{r['mode']:18} {r['threshold']:>5} {str(r['reused_correctly']) + '/' + str(r['same_meaning_pairs']):>10} "
              f"{str(r['reused_wrongly']) + '/' + str(r['different_pairs']):>13} {str(r['precision']):>10} {str(r['recall']):>7}")

    mode, threshold = choose_setting(rows)
    print(f"\nSuggested: CACHE_GUARD_MODE = \"{mode}\"  and  CACHE_THRESHOLD = {threshold}  (put both in config.py)")

    vocab, guards = setups[mode]
    missed, wrong = failures(pairs, similarities, threshold, vocab, guards)
    print(f"\nWith this setting, {len(wrong)} different pair(s) would be WRONGLY reused:")
    for item in wrong:
        print("  ", item)
    print(f"\nand {len(missed)} same-meaning pair(s) would be missed (first 8):")
    for item in missed[:8]:
        print("  ", item)

    os.makedirs("results", exist_ok=True)
    with open(config.CACHE_RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nFull table saved to", config.CACHE_RESULTS_PATH)


if __name__ == "__main__":
    main()
