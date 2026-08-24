# All-26 Physical Smoke Request

- Run ID: `all26_smoke_task1_seed104_20260824_100334`
- Status: `LAUNCH_PENDING`
- Producer commit: `6c24ef4`
- Task/seed: Task1, seed104
- Maximum environment steps: `300`
- Unix user: `hzhang061`
- Required gates: fresh one-GPU probe, fresh two-GPU formal-shape probe, then
  one two-GPU autonomous physical episode.
- Partition order: `acd_u`, `acd_ue`, `emergency_acd`; no explicit `-A`.
- Excluded nodes: `ACD1-8,ACD1-11,ACD1-31,ACD1-54`.
- Control: high-vlm-v2, VLA35999 checkpoint-local norm, two consecutive fresh
  VLM predictions before later prompt commit, no hold/release/anchor/oracle/GT
  replay.
- Scoring: RoboMemArena `cc156e519990ae43cf3b64281a548724f428fbbd`,
  scorer SHA-256
  `4eb949049b3175df01e8c632a6159a4b65bf9e2a667f6cbe612132ba5e7e0b99`.
- Preflight: submitting shell returned `whoami=hzhang061`; launcher readable;
  `records/` and `runs/` writable.
