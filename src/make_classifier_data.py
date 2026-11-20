


import csv
import json
import random

import config
from src.db import get_connection
from src.llm import generate

TARGET_PER_CLASS = 100

IN_SCOPE_STYLES = [
    "a short question",
    "a casual question written in lowercase, like a student typing quickly",
    "a how-to question",
    "a question that starts with 'why'",
    "a question that starts with 'what is'",
]

IN_SCOPE_PROMPT = """Read the passage and write ONE question that can be answered using this passage.
Write it as {style}. Do not mention "the passage".
Return JSON like: {{"question": "..."}}

Passage:
{text}
"""

OUT_PROMPT = """Write 5 different questions that a student might ask about: {topic}.
Return JSON like: {{"questions": ["...", "..."]}}"""


GENERAL_TOPICS = [
    "cricket and other sports", "cooking and recipes", "world geography and capitals",
    "history", "physics and chemistry", "movies and music", "health and fitness",
    "travel", "mathematics", "personal finance",
]


NEAR_TOPICS = [
    "CSS grid layout", "CSS animations and transitions", "React", "Node.js and npm",
    "SQL databases", "Git and GitHub", "Python programming",
    "JavaScript promises and async/await", "HTML video, audio and canvas elements",
    "Bootstrap and Tailwind CSS",
]


def draft_in_scope(existing_questions):
    rows = [{"question": q, "label": 1, "source": "eval_set"} for q in existing_questions]
    seen = {q.lower() for q in existing_questions}

    conn = get_connection()
    chunks = conn.execute(
        "SELECT text FROM chunks WHERE chunk_size = ?", (config.RAG_CHUNK_SIZE,)
    ).fetchall()
    random.shuffle(chunks)

    for (text,) in chunks[:200]:
        if len(rows) >= TARGET_PER_CLASS:
            break
        try:
            reply = generate(IN_SCOPE_PROMPT.format(style=random.choice(IN_SCOPE_STYLES), text=text),
                             json_mode=True)
            question = json.loads(reply)["question"].strip()
        except Exception as e:
            print("skip (bad reply):", e)
            continue
        if question.lower() in seen:
            continue
        seen.add(question.lower())
        rows.append({"question": question, "label": 1, "source": "llm_in_scope"})
        print(f"in-scope {len(rows)}: {question}")
    return rows


def draft_out_of_scope():
    rows = []
    seen = set()
    for kind, topics in [("llm_general", GENERAL_TOPICS), ("llm_near_topic", NEAR_TOPICS)]:
        for topic in topics:
            try:
                reply = generate(OUT_PROMPT.format(topic=topic), json_mode=True)
                questions = json.loads(reply)["questions"]
            except Exception as e:
                print("skip (bad reply):", e)
                continue
            for q in questions:
                if isinstance(q, str) and q.strip() and q.lower() not in seen:
                    seen.add(q.lower())
                    rows.append({"question": q.strip(), "label": 0, "source": kind})
            print(f"out-of-scope so far: {len(rows)} (after '{topic}')")
    return rows


def main():
    random.seed(config.RANDOM_SEED)

    with open(config.EVAL_PATH, newline="", encoding="utf-8") as f:
        eval_questions = [row["question"] for row in csv.DictReader(f)]

    rows = draft_in_scope(eval_questions) + draft_out_of_scope()

    with open(config.CLASSIFIER_CANDIDATES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    ones = sum(r["label"] for r in rows)
    print(f"\nSaved {len(rows)} rows ({ones} in-scope, {len(rows) - ones} out-of-scope)")
    print("Review them, then save the cleaned file as", config.CLASSIFIER_DATA_PATH)


if __name__ == "__main__":
    main()
