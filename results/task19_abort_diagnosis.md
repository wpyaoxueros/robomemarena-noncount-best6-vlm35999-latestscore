# Task19 SIGABRT diagnosis

- Attempt 1 on ACD1-22 completed one scored episode.
- Attempt 2 on ACD1-61 slowed from about one second to about twenty seconds per VLM inference around `t=190`, then exited with code 134 and no Python traceback.
- ACD1-31 showed the same severe VLM slowdown during Task4 and was excluded.
- The rollout parameters, VLM checkpoint, VLA 35999 checkpoint, norm, and official scorer were unchanged across these attempts.

Working hypothesis: the abort is node-specific runtime instability, not a deterministic Task19 evaluator error. The controlled test pins the unchanged Task19 20-episode run to the previously successful ACD1-22 node.
