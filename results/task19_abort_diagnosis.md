# Task19 SIGABRT diagnosis

- Attempt 1 on ACD1-22 completed one scored episode, but a later retry on that node became slow while unrelated non-Slurm inference processes were present.
- Attempt 2 on ACD1-61 slowed from about one second to about twenty seconds per VLM inference around `t=190`, then exited with code 134 and no Python traceback.
- ACD1-31 showed the same severe VLM slowdown during Task4 and was excluded.
- The rollout parameters, VLM checkpoint, VLA 35999 checkpoint, norm, and official scorer were unchanged across these attempts.

The same Task19 smoke completed on ACD1-40 during the current campaign with normal inference latency. The controlled retry therefore pins the unchanged Task19 20-episode run to ACD1-40. No model, norm, rollout, or scoring parameter is changed.
