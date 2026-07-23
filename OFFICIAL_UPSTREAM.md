# Official RoboMemArena scoring source

- Repository: `https://github.com/OpenHelix-Team/RoboMemArena`
- Branch checked: `main`
- Commit: `d9f83ac5182e25ad7f0a301a77a0b667f2392df1`
- Commit date: 2026-07-21
- `task2_26_reference_stage.py` SHA-256: `0ab5e19cb7b90844b86fe04a76facc0364af55f1e841c4754aa675404a318538`

The tracked `official_snapshot/evaluation_benchmark` files are exported from that exact commit. The large `libero_fork` runtime is excluded from Git and must be restored from the same commit before evaluation.

The evaluator fails if `task2_26_reference_stage.py` is unavailable. It never falls back to the legacy scorer.
