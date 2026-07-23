# VLM + VLA 35999 non-counting 20-episode reproduction

This package reproduces Task4, Task5, Task11, Task14, Task17, and Task19 with:

- the original Pi05 VLA checkpoint at step 35999;
- each task's retained best VLM and rollout parameters;
- the RoboMemArena official scoring code from commit `d9f83ac5182e25ad7f0a301a77a0b667f2392df1`;
- 20 episodes starting at seed 104.

The official scorer is loaded from `official_snapshot/evaluation_benchmark/scripts/task2_26_reference_stage.py`. The launcher fails if that file is missing; no fallback scorer is allowed.

Run one task through the validated borrowed account:

```bash
ssh zzhang510@localhost ./scripts/submit_task.sh 4 20 104
```

Run all six:

```bash
./scripts/submit_all.sh
```

Each output contains `run_manifest.env`, rollout logs, videos, `official_episodes.tsv`, `official_task_summary.tsv`, and `official_summary.json`.
