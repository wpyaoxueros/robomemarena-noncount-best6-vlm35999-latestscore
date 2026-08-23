# Native render abort lesson

## 已确认事实

- 崩溃是 evaluator 进程的原生 `SIGABRT`，栈停在 robosuite/MuJoCo
  `binding_utils.read_pixels -> mujoco.mjr_readPixels`。
- 崩溃 episode 没有 summary 数据行，不能记成模型失败。
- GT-replay 使用同一 VLM、LIBERO、双相机和图像保存路径跑过 300 step，说明
  VLM 推理和图像保存本身不足以触发崩溃。
- 同一 autonomous runtime 既有跑到 824/2500 step 的有效 episode，也有在
  250 step 崩溃的 episode，因此不能把 `step=250` 当作固定 horizon。
- Task2/3/4 三个并行 episode 都在 step 250、VLM call 51 崩溃，但当时的
  prompt、物理状态和任务不同，且 keyframe memory 为空。

## 之前为什么重复出错

之前提交到 Git 的是崩溃定位和无效样本处理，不是经过验证的修复。仅启用
`PYTHONFAULTHANDLER=1` 能显示 Python 栈，但不能说明是谁破坏了渲染上下文。
在没有补齐动作和 EGL 证据前直接扩跑 Task2/3/4，重复使用了同一个未修复
runtime。

## 后续强制门禁

1. 每个 VLA chunk 必须记录 action shape、finite、最大绝对值和实际执行 action。
2. 每次 `env.step` 前必须验证 action、MuJoCo qpos 和 qvel 都是有限值；不满足时
   干净退出并保存证据，不能继续进入 renderer。
3. manifest 必须记录 `CUDA_VISIBLE_DEVICES`、`SLURM_STEP_GPUS`、
   `MUJOCO_EGL_DEVICE_ID`、EGL device 数量和周期性 GPU memory。
4. 修复候选必须单变量测试：先测试 action/state guard，再测试 VLM CUDA 与
   MuJoCo EGL 分离到不同 GPU；不能把两项改动混在一次 run 里。
5. 任一候选先跑 autonomous physical 300-step smoke。没有连续通过门禁时，不得
   扩成多任务或 20ep。
6. `SIGABRT` 自动标为 invalid runtime attempt；保留日志、manifest、最后 action
   和最后物理状态，并换新 run ID。禁止盲目原样重跑。

## 当前状态

根因边界已缩小到“VLA action / MuJoCo physics / EGL rendering”链路，但 action
trace 尚未补齐，因此目前没有可以声称为已验证的最终修复。
