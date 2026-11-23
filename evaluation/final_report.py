"""Combine the repeat runs of the evaluation (<folder>/<strategy>-r1.json, -r2, ...) into one report.

Each strategy runs more than once because answers vary between identical runs (single comparison answers moved by
0.5). For every metric this reports the mean of the runs and the range (lowest, highest), per strategy and per question
type, then the costs, and which questions were unstable or wrong in some run. A folder may hold one strategy only.

No API calls: it only reads the result files. Run from the repo root:
python evaluation/final_report.py [folder, default evaluation/results/final3]  ->  <folder>/summary.json
"""

import json
import sys
from pathlib import Path
from statistics import mean

FINAL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "final3"
STRATEGIES = ("vector", "hybrid")
METRICS = [  # (key, label, higher is better)
    ("retrieval_hit_rate", "Correct source reached the model", True),
    ("retrieval_recall_at_5", "Recall@5", True),
    ("retrieval_recall_at_8", "Recall@8", True),
    ("retrieval_precision_at_5", "Precision@5", True),
    ("answer_accuracy", "Answer accuracy", True),
    ("groundedness", "Groundedness", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("relevance", "Relevance", True),
    ("expected_tool_used_rate", "Expected tool used", True),
    ("avg_latency_ms", "Average latency (ms)", False),
    ("avg_cost_usd", "Average cost per answer ($)", False),
]
PLAN_FIELDS = ("retrieval_recall_at_5", "answer_accuracy", "groundedness", "avg_latency_ms", "avg_cost_usd")


def spread(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None}
    average = mean(values)
    rounded = round(average) if all(isinstance(v, int) for v in values) else round(average, 4)  # ms stay whole
    return {"mean": rounded, "min": min(values), "max": max(values)}


def load(strategy: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FINAL.glob(f"{strategy}-r[0-9].json"))]


def main() -> None:
    runs = {s: load(s) for s in STRATEGIES}
    report = {"runs": {s: len(r) for s, r in runs.items()}, "overall": {}, "by_type": {}, "plan_fields": {},
              "questions": {}, "costs": {}}

    print(f"{'':34} {'vector (mean, range)':>24} {'hybrid (mean, range)':>24}")
    for key, label, _ in METRICS:
        row = {s: spread([r["summary"][key] for r in runs[s]]) for s in STRATEGIES}
        report["overall"][key] = row
        cells = [f"{row[s]['mean']} ({row[s]['min']}–{row[s]['max']})" for s in STRATEGIES]
        print(f"{label:34} {cells[0]:>24} {cells[1]:>24}")

    for s in STRATEGIES:
        report["plan_fields"][s] = {"strategy": s, **{k: report["overall"][k][s]["mean"] for k in PLAN_FIELDS}}
        report["costs"][s] = {
            "system_usd": round(sum(r["summary"]["avg_cost_usd"] * r["summary"]["questions"] for r in runs[s]), 4),
            "judge_usd": round(sum(r["summary"]["judge"]["cost_usd"] for r in runs[s]), 4),
        }

    print("\nBy question type, answer accuracy (mean of runs):")
    for qtype in runs["hybrid"][0]["summary"]["by_type"]:
        row = {s: spread([r["summary"]["by_type"][qtype]["answer_accuracy"] for r in runs[s]]) for s in STRATEGIES}
        n = runs["hybrid"][0]["summary"]["by_type"][qtype]["questions"]
        report["by_type"][qtype] = {"questions": n, **row,
                                    "recall_at_5": {s: spread([r["summary"]["by_type"][qtype]["retrieval_recall_at_5"]
                                                               for r in runs[s]])["mean"] for s in STRATEGIES}}
        print(f"  {qtype:15} n={n:2}  vector {row['vector']['mean']}  hybrid {row['hybrid']['mean']}")

    # Per question: correctness in every run, any unsupported claim, any fallback or failed translation.
    for s in STRATEGIES:
        for run in runs[s]:
            for q in run["questions"]:
                entry = report["questions"].setdefault(q["id"], {"type": q["type"], "question": q["question"]})
                cell = entry.setdefault(s, {"correctness": [], "hallucination": [], "warnings": 0, "errors": []})
                scores = q["scores"] or {}
                cell["correctness"].append(scores.get("correctness"))
                cell["hallucination"].append(scores.get("hallucination"))
                cell["warnings"] += bool(q.get("warnings"))
                if q["error"]:
                    cell["errors"].append(q["error"])

    print("\nQuestions not correct in every run (correctness per run):")
    for qid, entry in report["questions"].items():
        cells = {s: entry[s]["correctness"] for s in STRATEGIES if s in entry}
        if any(c != 1.0 for s in cells for c in cells[s]):
            print(f"  {qid:3} {entry['type']:14} vector {cells.get('vector')}  hybrid {cells.get('hybrid')} | "
                  f"{entry['question'][:55]}")
    flagged = {s: sum(e[s]["warnings"] for e in report["questions"].values() if s in e) for s in STRATEGIES}
    print(f"\nAnswers with a fallback warning: {flagged}")
    # Citations: of the chunks cited, the share about an expected film; "sent" is the same share of all chunks given to
    # the model. Runs made before the model cited its sources have none.
    report["citations"] = {
        s: {key: spread([r["summary"].get("citations", {}).get(key) for r in runs[s]])
            for key in ("answers_with_citation", "invalid", "precision", "sent_precision")}
        for s in STRATEGIES
    }
    print(f"Citations: {report['citations']}")
    print(f"Costs (all runs): {report['costs']}")

    out = FINAL / "summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
