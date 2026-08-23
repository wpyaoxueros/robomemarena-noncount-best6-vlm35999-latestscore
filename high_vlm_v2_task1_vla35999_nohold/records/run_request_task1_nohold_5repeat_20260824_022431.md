# Task1 no-hold five-repeat request

- Batch ID: `task1_nohold_5repeat_20260824_022431`.
- Task/seed: Task1 / seed104, five independent repeats.
- Repeat 0: `task1_seed104_nohold_native_trace_20260824_021105`, already running when the batch was requested.
- Repeats 1-4: launch in parallel after this request commit is pushed.
- VLM: copied high-vlm-v2 evaluator and local adapter snapshot.
- VLA: checkpoint 35999 with its checkpoint-local `robomemarena_fullvlm_v2_noflip_dataset_v2` norm.
- Control: synchronous VLM prompt to VLA actions; no hold, release, anchor, oracle prompt injection, GT replay or trajectory-only control.
- Resources: independent two-GPU formal allocation per repeat, with fresh borrowed-account probe gates handled by each immutable launch request.
- Required evidence per valid repeat: manifest, evaluator/server logs, prompt trace, episode summary, main video and wrist video.
- Invalid-run rule: a native abort without an episode summary is recorded but excluded from the five-repeat success denominator.
