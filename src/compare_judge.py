


import csv
import json

from sklearn.metrics import cohen_kappa_score

import config


def agreement_rate(human, judge):
    return sum(1 for h, j in zip(human, judge) if h == j) / len(human)


def kappa(human, judge):
    if len(set(human)) < 2 and len(set(judge)) < 2:
        return None
    value = cohen_kappa_score(human, judge)
    return None if value != value else round(float(value), 3)


def load_pairs(judge_path, human_path, metric):
    with open(judge_path, newline="", encoding="utf-8") as f:
        judge_rows = {r["question"]: r for r in csv.DictReader(f)}
    with open(human_path, newline="", encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))

    pairs = []
    for h in human_rows:
        j = judge_rows.get(h["question"])
        h_label, j_label = h[f"human_{metric}"].strip(), (j or {}).get(f"judge_{metric}", "").strip()
        if h_label in ("0", "1") and j_label in ("0", "1"):
            pairs.append({"question": h["question"], "answer": h["answer"], "notes": h["notes"],
                          "human": int(h_label), "judge": int(j_label)})
    return pairs


def summarize(pairs):
    human = [p["human"] for p in pairs]
    judge = [p["judge"] for p in pairs]
    both = lambda a, b: sum(1 for h, j in zip(human, judge) if h == a and j == b)
    return {
        "labelled_answers": len(pairs),
        "agreement": round(agreement_rate(human, judge), 3),
        "cohen_kappa": kappa(human, judge),
        "human_yes_rate": round(sum(human) / len(human), 3),
        "judge_yes_rate": round(sum(judge) / len(judge), 3),

        "confusion_human_vs_judge": [[both(0, 0), both(0, 1)], [both(1, 0), both(1, 1)]],
    }


def main():
    summary = {}
    disagreements = []
    for metric in ["faithful", "relevant"]:
        pairs = load_pairs(config.ANSWER_EVAL_PATH, config.HUMAN_SHEET_PATH, metric)
        if not pairs:
            print(f"No labelled rows for '{metric}'. Fill human_{metric} in {config.HUMAN_SHEET_PATH} first.")
            continue
        summary[metric] = summarize(pairs)
        for p in pairs:
            if p["human"] != p["judge"]:
                disagreements.append({"metric": metric, **p})
        print(f"\n{metric}: {json.dumps(summary[metric])}")

    with open(config.JUDGE_AGREEMENT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    if disagreements:
        with open(config.JUDGE_DISAGREEMENTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)
    print(f"\n{len(disagreements)} disagreements saved to {config.JUDGE_DISAGREEMENTS_PATH}")


if __name__ == "__main__":
    main()
