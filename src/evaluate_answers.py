


import csv
import json
import os
import re
import sys

import config
from src.classifier import ScopeClassifier
from src.llm import generate
from src.metrics import is_relevant
from src.rag import answer_question, build_context
from src.retrieval import DenseRetriever

JUDGE_PROMPT = """You are checking an answer written by an assistant.

Sources:
{sources}

Question: {question}
Answer: {answer}

Return JSON with two keys:
"faithful": "yes" if EVERY fact in the answer is supported by the sources, otherwise "no"
"relevant": "yes" if the answer actually addresses the question, otherwise "no"
"""

STATEMENT_PROMPT = """Sources:
{sources}

Statement: {statement}

Is the statement supported by the sources?
Return JSON like: {{"supported": "yes"}} or {{"supported": "no"}}
"""

DEFINITIONS = {
    "false_refusal_rate": "refused in-scope questions / all questions",
    "faithfulness": "answers where the judge says every fact is supported / answered questions the judge could grade",
    "answer_relevance": "answers the judge says address the question / answered questions the judge could grade",
    "hallucination_rate": "sentences the judge says are NOT supported / all sentences checked (answered questions only)",
    "citation_correctness": "answers whose cited chunks contain the gold evidence passage / answered questions",
    "citation_present_rate": "answers with at least one [n] citation / answered questions",
}


def judge_generate(prompt, json_mode=False):
    return generate(prompt, json_mode=json_mode, model=config.JUDGE_MODEL)


def to_bool(value):
    text = str(value).strip().lower()
    if text in ("yes", "true"):
        return True
    if text in ("no", "false"):
        return False
    return None


def judge_answer(question, answer, context, llm=judge_generate):
    prompt = JUDGE_PROMPT.format(sources=context, question=question, answer=answer)
    try:
        data = json.loads(llm(prompt, json_mode=True))
    except Exception:
        return None, None
    return to_bool(data.get("faithful")), to_bool(data.get("relevant"))


def split_statements(answer):
    answer = re.sub(r"\s*\[\d+\]", "", answer)
    pieces = re.split(r"(?<=[.!?])\s+", answer)
    return [p.strip() for p in pieces if len(p.split()) >= 4]


def count_unsupported(answer, context, llm=judge_generate):
    unsupported, checked = 0, 0
    for statement in split_statements(answer):
        prompt = STATEMENT_PROMPT.format(sources=context, statement=statement)
        try:
            supported = to_bool(json.loads(llm(prompt, json_mode=True)).get("supported"))
        except Exception:
            continue
        if supported is None:
            continue
        checked += 1
        if not supported:
            unsupported += 1
    return unsupported, checked


def citation_correct(result, evidence):
    return any(is_relevant(chunk["text"], evidence) for chunk in result["sources"])


def evaluate_one(item, retriever, classifier=None, answer_llm=generate, judge_llm=judge_generate):
    result = answer_question(item["question"], retriever, llm=answer_llm, classifier=classifier)
    context = build_context(result["retrieved"])

    row = {
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "evidence": item["evidence"],
        "answer": result["answer"],
        "refused": result["refused"],
        "reason": result["reason"],
        "best_score": round(result["best_score"], 3),
        "scope_probability": result["scope_probability"],
        "cited_urls": " | ".join(c["source_url"] for c in result["sources"]),
        "evidence_in_retrieved": any(is_relevant(c["text"], item["evidence"]) for c in result["retrieved"]),
        "judge_faithful": None, "judge_relevant": None,
        "unsupported_statements": 0, "total_statements": 0,
        "has_citation": None, "citation_correct": None,
        "cited_text": " || ".join(c["text"] for c in result["sources"])[:1500],
        "retrieved_context": context,
    }

    if not result["refused"]:
        row["judge_faithful"], row["judge_relevant"] = judge_answer(
            item["question"], result["answer"], context, judge_llm)
        row["unsupported_statements"], row["total_statements"] = count_unsupported(
            result["answer"], context, judge_llm)
        row["has_citation"] = len(result["sources"]) > 0
        row["citation_correct"] = citation_correct(result, item["evidence"])
    return row


def ratio(numerator, denominator):
    value = round(numerator / denominator, 3) if denominator else None
    return {"value": value, "numerator": numerator, "denominator": denominator}


def compute_metrics(rows):
    answered = [r for r in rows if not r["refused"]]
    graded_f = [r for r in answered if r["judge_faithful"] is not None]
    graded_r = [r for r in answered if r["judge_relevant"] is not None]
    return {
        "questions": len(rows),
        "answered": len(answered),
        "false_refusal_rate": ratio(len(rows) - len(answered), len(rows)),
        "faithfulness": ratio(sum(r["judge_faithful"] for r in graded_f), len(graded_f)),
        "answer_relevance": ratio(sum(r["judge_relevant"] for r in graded_r), len(graded_r)),
        "hallucination_rate": ratio(sum(r["unsupported_statements"] for r in answered),
                                    sum(r["total_statements"] for r in answered)),
        "citation_correctness": ratio(sum(1 for r in answered if r["citation_correct"]), len(answered)),
        "citation_present_rate": ratio(sum(1 for r in answered if r["has_citation"]), len(answered)),
        "definitions": DEFINITIONS,
    }


def badness(row):
    if row["refused"]:
        return 3
    score = 0
    if row["judge_faithful"] is False:
        score += 3
    if row["judge_relevant"] is False:
        score += 2
    if row["citation_correct"] is False:
        score += 1
    return score + min(row["unsupported_statements"], 3)


def worst_answers(rows, n=5):
    return sorted(rows, key=badness, reverse=True)[:n]


def diagnosis_hint(row):
    if not row["evidence_in_retrieved"]:
        return "retrieval miss: the evidence passage was not in the top-5 chunks"
    if row["refused"]:
        return f"refused ({row['reason']}) although the evidence WAS retrieved"
    return "generation problem: the evidence was retrieved but the answer is still weak"


def write_worst_md(rows, path):
    lines = ["# Five worst answers (error analysis)\n",
             "Ranked by a simple points system (refused = 3, unfaithful = 3, irrelevant = 2, "
             "wrong citation = 1, plus unsupported sentences). Fill in the last two lines of each.\n"]
    for i, row in enumerate(worst_answers(rows), 1):
        lines += [
            f"\n## {i}. {row['question']}\n",
            f"- **Expected answer:** {row['expected_answer']}",
            f"- **Actual answer:** {row['answer']}",
            f"- **Badness score:** {badness(row)} | refused: {row['refused']} ({row['reason']}) | "
            f"judge faithful: {row['judge_faithful']} | judge relevant: {row['judge_relevant']} | "
            f"citation correct: {row['citation_correct']}",
            f"- **Automatic hint:** {diagnosis_hint(row)}",
            f"- **Retrieved context:**\n\n```\n{row['retrieved_context'][:1800]}\n```",
            "- **Reason for failure:** (write after reading the context)",
            "- **Proposed improvement:** (write)",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(v) for k, v in row.items()})


def write_human_sheet(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "expected_answer", "answer", "cited_text",
            "human_faithful", "human_relevant", "notes"])
        writer.writeheader()
        for row in rows:
            if not row["refused"]:
                writer.writerow({"question": row["question"], "expected_answer": row["expected_answer"],
                                 "answer": row["answer"], "cited_text": row["cited_text"],
                                 "human_faithful": "", "human_relevant": "", "notes": ""})


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with open(config.EVAL_PATH, newline="", encoding="utf-8") as f:
        eval_set = list(csv.DictReader(f))[:limit]

    retriever = DenseRetriever(config.RAG_CHUNK_SIZE)
    classifier = ScopeClassifier() if os.path.exists(config.CLASSIFIER_PATH) else None
    print("Classifier:", "loaded" if classifier else "not found (similarity check only)")

    rows = []
    for i, item in enumerate(eval_set, 1):
        print(f"[{i}/{len(eval_set)}] {item['question']}")
        try:
            rows.append(evaluate_one(item, retriever, classifier))
        except Exception as e:
            print("   failed:", e)

    os.makedirs("results", exist_ok=True)
    write_rows(rows, config.ANSWER_EVAL_PATH)
    write_human_sheet(rows, config.HUMAN_SHEET_PATH)
    write_worst_md(rows, config.WORST_PATH)
    metrics = compute_metrics(rows)
    with open(config.ANSWER_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Answer quality ===")
    for name in DEFINITIONS:
        m = metrics[name]
        print(f"{name}: {m['value']}  ({m['numerator']} / {m['denominator']})")
    print("\nSaved:", config.ANSWER_EVAL_PATH, config.ANSWER_METRICS_PATH,
          config.HUMAN_SHEET_PATH, config.WORST_PATH, sep="\n  ")


if __name__ == "__main__":
    main()
