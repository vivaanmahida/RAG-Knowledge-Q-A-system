


import csv
import json
import random

import config
from src.db import get_connection
from src.llm import generate
from src.metrics import is_relevant

TARGET = 50
MAX_TRIES = 100

PROMPT = """Read the passage and write ONE question that can be answered using only this passage.
Do not write "according to the passage" in the question.
Return JSON with exactly these keys:
"question": the question
"answer": a short answer, taken from the passage
"evidence": copy 6 to 12 consecutive words EXACTLY as they appear in the passage, which prove the answer

Passage:
{text}
"""


def main():
    random.seed(42)
    conn = get_connection()
    rows = conn.execute(
        "SELECT source_url, title, text FROM chunks WHERE chunk_size = 500"
    ).fetchall()
    random.shuffle(rows)

    good = []
    for tries, (url, title, text) in enumerate(rows[:MAX_TRIES]):
        if len(good) >= TARGET:
            break
        try:
            reply = generate(PROMPT.format(text=text), json_mode=True)
            item = json.loads(reply)
            question, answer, evidence = item["question"], item["answer"], item["evidence"]
        except Exception as e:
            print("skip (bad reply):", e)
            continue


        if not is_relevant(text, evidence):
            print("skip (evidence not found in text)")
            continue

        good.append({"question": question, "expected_answer": answer,
                     "evidence": evidence, "source_url": url, "title": title})
        print(f"{len(good)}: {question}")

    with open(config.EVAL_CANDIDATES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "expected_answer", "evidence", "source_url", "title"]
        )
        writer.writeheader()
        writer.writerows(good)

    print(f"\nSaved {len(good)} candidates to {config.EVAL_CANDIDATES_PATH}")
    print("Now review them, then save 30-50 good ones as", config.EVAL_PATH)


if __name__ == "__main__":
    main()
