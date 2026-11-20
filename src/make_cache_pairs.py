


import csv
import json
import random

import config
from src.llm import generate

HOW_MANY = 30

SAME_PROMPT = """Rewrite this question in different words. Keep exactly the same meaning and keep the
same technical terms (the same HTML tag, CSS property or JavaScript keyword).
Question: "{q}"
Return JSON like: {{"question": "..."}}"""

DIFFERENT_PROMPT = """Rewrite this question so it asks about a DIFFERENT but related thing
(for example a different HTML tag, CSS property or JavaScript keyword). Use almost the same wording.
Question: "{q}"
Return JSON like: {{"question": "..."}}"""


def rewrite(prompt_template, question):
    try:
        return json.loads(generate(prompt_template.format(q=question), json_mode=True))["question"].strip()
    except Exception as e:
        print("skip (bad reply):", e)
        return None


def main():
    random.seed(config.RANDOM_SEED)
    with open(config.EVAL_PATH, newline="", encoding="utf-8") as f:
        questions = [row["question"] for row in csv.DictReader(f)]
    chosen = random.sample(questions, min(HOW_MANY, len(questions)))

    rows = []
    for q in chosen:
        same = rewrite(SAME_PROMPT, q)
        different = rewrite(DIFFERENT_PROMPT, q)
        if same:
            rows.append({"question_a": q, "question_b": same, "label": 1, "kind": "paraphrase"})
        if different:
            rows.append({"question_a": q, "question_b": different, "label": 0, "kind": "hard_negative"})
        print(f"{len(rows)} pairs so far")


    for _ in range(20):
        a, b = random.sample(questions, 2)
        rows.append({"question_a": a, "question_b": b, "label": 0, "kind": "random_negative"})

    with open(config.CACHE_PAIRS_CANDIDATES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_a", "question_b", "label", "kind"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} pairs to {config.CACHE_PAIRS_CANDIDATES_PATH}")
    print("Check that every label-1 pair really means the same, and every label-0 pair really differs.")
    print("Then save the cleaned file as", config.CACHE_PAIRS_PATH)


if __name__ == "__main__":
    main()
