# Task1 high_vlm_v2 + VLA35999 no-hold evaluation

This directory evaluates Task1 with a byte-for-byte copy of the VLM evaluation logic from:

```text
/data/user/jwen341/openpi_rm/vlm_ft
```

The source directory above is never modified. Copied code is under `source/`, large runtime assets are copied into ignored `runtime_assets/`, and generated rollouts are written to ignored `runs/`.

## Frozen evaluation contract

- Task: `1`
- Seed: `104`
- VLM architecture: `high_vlm_v2`
- VLM checkpoint: copied `final_adapter` from the documented 26-task run
- VLA config: `pi05_libero_robomemarena_fullvlm_v2_noflip_dataset`
- VLA checkpoint step: `35999`
- Action source: `vla`
- VLM mode: synchronous, queried every 5 environment steps
- VLA replanning: 5 actions
- Recent VLM frames: 5 dual-camera timesteps
- Historical keyframes: enabled, maximum 8
- Maximum rollout length: 2500 environment steps
- Video: main and wrist views

There is no hold/release logic, no pose/object anchor, no oracle-generated next prompt, and no GT action replay. The copied evaluator directly applies the VLM's current primitive as the prompt passed to the VLA websocket client.

The scoring and semantic tracker are the ones embedded in the copied evaluator. This experiment is specifically a compatibility test of `jwen341`'s VLM evaluation path; it is not relabeled as a separate latest-upstream RoboMemArena evaluation.

## Layout

```text
source/                  exact copied evaluator/runtime source
scripts/                 source verification, probes, and launchers
records/                 lightweight provenance and result receipts
runtime_assets/          copied adapter and LIBERO source, ignored by Git
runs/                    logs, traces, summaries, and videos, ignored by Git
SOURCE_MANIFEST.sha256   hashes of committed copied source
```

## Reproduce

1. Copy and verify runtime assets:

```bash
scripts/copy_runtime_assets.sh
scripts/verify_source_snapshot.sh --against-source
scripts/audit_nohold_contract.sh
```

2. From the authorized `hzhang061` login shell, launch the fresh probes and the two-GPU evaluation:

```bash
tmux new-session -d -s task1_highvlmv2_nohold_$(date +%Y%m%d_%H%M%S) \
  "scripts/launch_hzhang061_task1_seed104.sh"
```

The launcher refuses to run under a different Unix user, performs a fresh one-GPU probe and a fresh two-GPU request-shape probe, then starts the formal evaluation through `srun`.

