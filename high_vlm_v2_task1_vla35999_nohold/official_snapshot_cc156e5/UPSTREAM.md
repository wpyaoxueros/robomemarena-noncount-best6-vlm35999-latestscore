# RoboMemArena Official Scoring Snapshot

- Repository: `https://github.com/OpenHelix-Team/RoboMemArena.git`
- Branch observed: `main`
- Commit: `cc156e519990ae43cf3b64281a548724f428fbbd`
- Retrieved: `2026-08-24` through `all_proxy=socks5://localhost:9632`
- Scope: 26 top-level benchmark BDDL files, the shared Task1-26 stage
  scorer, its persistent pour counter, and the OpenPI minimal-runtime
  `eval_common.py`, `robocerebra_adapter.py`, and `task_prompts.py` imported by
  the existing high-vlm-v2 reference path.

This directory is append-only and does not replace the historical
`official_snapshot/` tree. Runtime code must verify `SHA256SUMS` and fail
closed if any required file is missing or changed.
