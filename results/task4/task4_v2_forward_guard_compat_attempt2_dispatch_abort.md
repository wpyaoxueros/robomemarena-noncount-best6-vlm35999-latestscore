# Task4 v2 Attempt 2: Dispatcher Abort

The queued Slurm probe `433848` was cancelled before resource allocation by
the outer SSH process. No tmux session, VLA server, evaluator, episode or
video was created. This is a dispatcher-only failure and is excluded from
Task4 evaluation results.

`submit_task4_v2_forward_guard_compat_retry.sh` moves both the fresh probe and
the actual run into one persistent tmux session so a queued probe cannot be
cancelled when the submitting SSH command exits.
