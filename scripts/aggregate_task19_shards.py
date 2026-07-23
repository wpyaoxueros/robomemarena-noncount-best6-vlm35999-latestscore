#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCORE_RE = re.compile(
    r"\[OFFICIAL_SCORE\] task=19 average_score_pct=([0-9.]+) "
    r"stage_success=([01]) goal_success=([01]) stage_done_json=(\{.*\})"
)
SEED_RE = re.compile(r"seed(\d+)")
EP_RE = re.compile(r"ep(\d+)$")


def parse_run(run: Path) -> list[dict]:
    seed_match = SEED_RE.search(run.name)
    if seed_match is None:
        raise ValueError(f"cannot infer seed start from {run}")
    seed_start = int(seed_match.group(1))
    rows = []
    for log in sorted((run / "task19").glob("ep*/sync_vlm.log")):
        ep_match = EP_RE.match(log.parent.name)
        if ep_match is None:
            continue
        matches = SCORE_RE.findall(log.read_text(encoding="utf-8", errors="ignore"))
        if not matches:
            continue
        score, stage_success, goal_success, stage_json = matches[-1]
        ep = int(ep_match.group(1))
        rows.append(
            {
                "task_id": 19,
                "ep": ep,
                "seed": seed_start + ep,
                "score_pct": float(score),
                "tsr_success": bool(int(stage_success)),
                "stage_success": bool(int(stage_success)),
                "goal_success": bool(int(goal_success)),
                "stage_done": json.loads(stage_json),
                "log": str(log.resolve()),
                "run": str(run.resolve()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    by_seed: dict[int, dict] = {}
    for run in args.run:
        for row in parse_run(run):
            seed = row["seed"]
            if seed in by_seed:
                raise ValueError(f"duplicate seed {seed}: {by_seed[seed]['log']} and {row['log']}")
            by_seed[seed] = row

    expected = set(range(104, 124))
    actual = set(by_seed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"unexpected seeds: {extra}")
    if missing and not args.allow_incomplete:
        raise ValueError(f"missing seeds: {missing}")

    rows = [by_seed[seed] for seed in sorted(actual)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "official_episodes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("task_id\tep\tseed\tscore_pct\ttsr_success\tstage_success\tgoal_success\tlog\n")
        for index, row in enumerate(rows):
            handle.write(
                f"19\t{index}\t{row['seed']}\t{row['score_pct']:.1f}\t"
                f"{'Y' if row['tsr_success'] else 'N'}\t{'Y' if row['stage_success'] else 'N'}\t"
                f"{'Y' if row['goal_success'] else 'N'}\t{row['log']}\n"
            )

    n = len(rows)
    summary = {
        "task_id": 19,
        "num_trials": n,
        "seed_start": 104,
        "average_score_pct": sum(row["score_pct"] for row in rows) / max(1, n),
        "tsr_success_rate_pct": 100 * sum(row["tsr_success"] for row in rows) / max(1, n),
        "stage_success_rate_pct": 100 * sum(row["stage_success"] for row in rows) / max(1, n),
        "goal_success_rate_pct": 100 * sum(row["goal_success"] for row in rows) / max(1, n),
        "missing_seeds": missing,
        "source_runs": [str(run.resolve()) for run in args.run],
    }
    (args.output / "official_summary.json").write_text(
        json.dumps({"episodes": rows, "tasks": [summary]}, indent=2), encoding="utf-8"
    )
    with (args.output / "official_task_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "task_id\tnum_trials\tseed_start\taverage_score_pct\ttsr_success_rate_pct\t"
            "stage_success_rate_pct\tgoal_success_rate_pct\n"
        )
        handle.write(
            f"19\t{n}\t104\t{summary['average_score_pct']:.1f}\t"
            f"{summary['tsr_success_rate_pct']:.1f}\t{summary['stage_success_rate_pct']:.1f}\t"
            f"{summary['goal_success_rate_pct']:.1f}\n"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
