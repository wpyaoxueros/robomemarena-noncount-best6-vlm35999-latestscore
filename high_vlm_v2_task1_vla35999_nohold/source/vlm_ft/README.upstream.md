# 高层 VLM：轨迹切分、训练与回放测评

本目录实现 RoboMemArena 长任务中的高层 VLM。它不直接输出机械臂动作，而是在完整任务执行过程中，根据双相机时序图像判断当前时间步应交给低层控制器的 primitive，并维护少量视觉关键帧记忆。

当前工程包含两个模型版本：

| 版本 | 主要输出 | 训练范围 | 当前推荐用途 |
|---|---|---|---|
| `high_vlm_v1` | `current_primitive`、`keyframe_positions` | Task 1/17/22 | 纯轨迹切分基线和消融 |
| `high_vlm_v2` | V1 输出 + 当前 primitive 的连续进度 | 26 个任务 | 当前主模型 |

详细实验过程、数据统计、作业号和历史故障见仓库根目录的 [vlm_training_log.md](../vlm_training_log.md)。本文重点说明工程结构和可复现流程。

## 1. 高层 VLM 在三系统中的位置

```text
仿真环境双相机观测
        │
        ├── 最近 5 个连续 timestep（每步 agent-view + wrist）
        └── 最多 8 个历史关键帧（每帧 agent-view + wrist）
                         │
                         ▼
                    high-level VLM
                         │
              current_primitive + keyframes
                         │
                         ▼
                  low-level VLA/controller
                         │
                         ▼
                       action
```

监督目标是“原始演示轨迹在当前时间步分配的 primitive”，不是根据视觉语义提前猜测下一任务。统一 system prompt 的核心句是：

```text
Predict the low-level controller primitive assigned to the current
timestep in the demonstrated trajectory segmentation. Use the latest
timestep as the current state.
```

因此，训练标签和 SCNA 指标都以真实 HDF5 子轨迹边界为准。

## 2. 输入与输出契约

### 2.1 图像输入

每次调用包含：

- 最近窗口 `R1...R5`：同一条完整轨迹中连续 5 个 timestep，允许跨 HDF5 子轨迹边界；`R5` 是当前状态。
- 每个 timestep 有两个视角：`agentview_rgb` 和 `eye_in_hand_rgb`。
- 历史窗口 `H1...Hn`：严格早于 `R1` 的历史关键帧，按时间排序，最多 8 个；不在子轨迹边界清空。
- Prompt 显式写出 `H/R`、`global_t`、相对时间 `dt`、相机名称和 `CURRENT` 标记。

训练和26任务推理共同使用 [prompt_contract_26.py](./prompt_contract_26.py)，不要在评测脚本中复制一份不同的 prompt。

### 2.2 V1 生成输出

V1 只生成严格 JSON：

```json
{
  "current_primitive": "place cookies into basket",
  "keyframe_positions": [3]
}
```

- `current_primitive` 必须逐字匹配该任务的合法 primitive。
- `keyframe_positions` 是 `R1...R5` 内关键帧的 1-based 位置，若没有则为 `[]`。
- 任务描述、场景描述、合法 primitive 和 HDF5 stem 映射来自校正后的26任务配置：
  `eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json`。
- 配置以真实轨迹内容为准，不能只依赖官网任务名称。

### 2.3 V2 进度输出

V2 保持相同 JSON，同时增加输入专用特殊 token：

```text
system + user(images) + assistant generation header
+ <|progress_query|> + assistant JSON
```

进度头读取 `<|progress_query|>` 在语言模型最后一层、最终 RMSNorm 后的 hidden state：

```text
4096 → 256 → GELU → Dropout(0.1) → 1 → sigmoid
```

输出是当前 primitive 内的归一化进度 `[0, 1]`。它不写入 assistant JSON，也不参与文本 decode。

进度真值严格由 R5 所属 HDF5 子轨迹计算：

```text
progress_target = R5_local_step / max(segment_action_count - 1, 1)
```

特殊 token 必须满足：

- tokenizer 中恰好编码为一个 token；当前训练版本 ID 为 `151669`。
- 位于 JSON 第一个 token 之前，LM label 为 `-100`。
- 推理时加入 `suppress_tokens`，避免模型生成它。
- decode 必须按完整输入长度裁掉输入前缀，只解码新增 continuation。

这些约束统一实现在 [high_vlm_v2_components.py](./high_vlm_v2_components.py) 中。

## 3. 两个模型版本如何训练

### 3.1 `high_vlm_v1`：纯生成 LoRA

V1 的推荐权重不是早期的 `three_tasks_strict_scna_v1` 数据实验，而是使用改进后的历史条件平衡数据训练出的三任务模型。历史目录中带 `v2` 指的是“数据方案 V2”，模型架构仍叫 `high_vlm_v1`。

| 项目 | 配置 |
|---|---|
| 基座 | Qwen3-VL-8B-Instruct |
| 任务 | Task 1、17、22 |
| 训练数据 | 72,000 条 balanced view，每任务 24,000 条 |
| LoRA | `r=16, alpha=32, dropout=0.05` |
| LoRA target | 文本塔 `q/k/v/o`、`gate/up/down` |
| 视觉塔 | 冻结 |
| Loss | assistant JSON token 1:1 交叉熵 |
| GPU | 单机 8×H100 |
| batch | 每卡 1，梯度累积 2，有效 batch 16 |
| 训练长度 | 1 epoch，约 4,500 optimizer steps |
| LR | `1e-4`，cosine，warmup 5% |
| 最大长度 | 4096 token |
| 验证/保存 | 每 500 step |

训练入口：

```bash
cd /data/user/jwen341/openpi_rm

OUTPUT_DIR=/data/user/jwen341/openpi_rm/output/my_high_vlm_v1 \
sbatch vlm_ft/slurm_train_three_task_lora_v2_8gpu.sh
```

已完成权重：

```text
output/vlm_three_task_lora_v2_8gpu_xiangqim_20260812/final_adapter/
```

独立 GT 回放中，Task 1/17/22 的微平均 SCNA@0 为 82.0%，SCNA@2 为 100%。

### 3.2 `high_vlm_v2`：LoRA + 进度头

V2 在 V1 的生成目标之外联合训练进度回归：

```text
total_loss = lm_loss + 0.1 × progress_loss
progress_loss = SmoothL1(sigmoid(progress_logit), progress_target, beta=0.1)
```

primitive 和 keyframe JSON token 保持 1:1 权重。V2 会拒绝非 `1.0` 的 `--primitive-loss-weight`，防止再次引入 primitive 字段过度加权。

26任务正式训练配置：

| 项目 | 配置 |
|---|---|
| 基座 | Qwen3-VL-8B-Instruct |
| 任务 | Task 1–26，共157个实际 primitive 标签 |
| 训练数据 | 624,000 条 balanced view，每任务 24,000 条 |
| LoRA | `r=32, alpha=64, dropout=0.05` |
| 额外训练参数 | progress head + 新增 query token embedding |
| 可训练参数 | 88,347,137 |
| 视觉塔 | 冻结 |
| GPU | 单机 8×H100 |
| batch | 每卡 1，梯度累积 4，有效 batch 32 |
| 训练长度 | 1 epoch，19,500 optimizer steps |
| LR | `5e-5`，cosine，warmup 3%，weight decay 0.01 |
| 最大长度 | 4096 token |
| 验证集 | 分层抽取 1,560 条 |
| 验证/保存 | 每 2,000 step |

正式提交：

```bash
cd /data/user/jwen341/openpi_rm

OUTPUT_DIR=/data/user/jwen341/openpi_rm/output/my_high_vlm_v2_26tasks \
sbatch vlm_ft/slurm_train_high_vlm_v2_26tasks_8gpu.sh
```

训练前只检查数据、processor、token 位置和空权重结构：

```bash
export PYTHONPATH=/data/user/jwen341/openpi_rm
export TRANSFORMERS_NO_TF=1 USE_TF=0

/data/user/hlei573/openpi_inference/.venv/bin/python \
  vlm_ft/train_high_vlm_v2.py \
  --train-data vlm_ft/datasets/high_vlm_v2_26tasks/swift_compiled_data_train_balanced.jsonl \
  --eval-data vlm_ft/datasets/high_vlm_v2_26tasks/swift_compiled_data_val.jsonl \
  --output-dir /tmp/high_vlm_v2_preflight \
  --lora-r 32 --lora-alpha 64 \
  --preflight-only
```

已完成的正式训练：

```text
output/high_vlm_v2_26tasks_lora_r32_8gpu_499178/final_adapter/
```

- 训练作业：`499178`，19,500/19,500 steps，耗时 10:48:23。
- 汇总 train loss：0.014197。
- 最佳 checkpoint：step 14,000，`eval_loss=0.00555848`，`eval_progress_mae=0.0254549`。
- `final_adapter` 已通过 `load_best_model_at_end` 恢复 step 14,000，而不是直接保存最后一步。
- 26任务独立 GT 回放已完成：260条 test 轨迹、1,220个边界，微平均 SCNA@0/SCNA@2 为82.95%/94.10%，Progress MAE 为0.09544；完整逐任务结果见 [vlm_training_log.md](../vlm_training_log.md)。

## 4. 26任务训练数据如何构建

原始数据：

```text
/data/user/jwen341/dataset/robomemarena_fullvlm_v2_official_remote_20260711_raw
```

每个任务有100条完整演示轨迹。构建器按 HDF5 文件名中的 `order` 拼接子轨迹，形成任务级全局时间轴，然后在 `t=5,10,15,...` 上生成查询样本。

### 4.1 数据划分

每个任务独立按 seed 排序切分：

| Split | 每任务轨迹数 | 26任务轨迹数 |
|---|---:|---:|
| train | 80 | 2,080 |
| val | 10 | 260 |
| test | 10 | 260 |

train/val/test 不共享 seed。独立回放必须使用 test split 的260条轨迹。

### 4.2 历史和关键帧采样

常规端点使用确定性的历史条件混合：

- 40% 无历史 `none`；
- 30% 部分历史 `partial`；
- 30% 最多8帧完整历史 `full`。

部分历史会随机选择最近或更分散的过去关键帧，但绝不插入未来帧。若最近5帧内存在标注关键帧，该端点会生成所有可行历史条件，供后续关键帧正样本平衡。

样本位置分为：

- `regular`：普通子轨迹内部；
- `near_boundary`：靠近真实 HDF5 边界；
- `scna_k0`：边界后第一次位于5步调用网格上的查询。

### 4.3 平衡训练视图

原始唯一数据和最终训练视图：

| 数据 | 数量 |
|---|---:|
| train unique | 466,498 |
| val unique | 58,446 |
| test unique | 58,435 |
| balanced train | 624,000 |
| balanced train unique qid | 207,219（33.21%） |

平衡优先级是：

1. 26个任务等权，每任务24,000条；
2. 每个任务内部 primitive 等权；
3. 每任务关键帧正/负样本为8,400/15,600，即35%/65%；
4. 在已有池内轮转 `sample_type × history_condition`。

平衡文件采用有放回抽样，因此重复率较高；正式训练只使用1 epoch。

### 4.4 完整构建命令

以下命令会生成约558万张 JPEG，耗时和磁盘占用都很大。不要在已有数据目录上随意使用 `--overwrite`。

```bash
cd /data/user/jwen341/openpi_rm
export PYTHONPATH=/data/user/jwen341/openpi_rm/vlm_ft
PY=/data/user/hlei573/openpi_inference/.venv/bin/python
DATA=vlm_ft/datasets/high_vlm_v2_26tasks

# 1. 从 HDF5 构造唯一 JSONL、manifest 和边界审计
$PY vlm_ft/build_high_vlm_v2_26tasks_dataset.py \
  --source /data/user/jwen341/dataset/robomemarena_fullvlm_v2_official_remote_20260711_raw \
  --output "$DATA"

# 2. 生成624,000条平衡训练视图
$PY vlm_ft/make_balanced_train_v2_26tasks.py \
  "$DATA/swift_compiled_data_train.jsonl" \
  "$DATA/swift_compiled_data_train_balanced.jsonl" \
  --per-task-samples 24000 \
  --keyframe-positive-ratio 0.35 \
  --seed 20260813

# 3. 从 HDF5 原子物化256×256、quality=95的双相机 JPEG
$PY vlm_ft/materialize_v2_26tasks_images.py "$DATA" --workers 32

# 4. 全量检查时序、标签、进度、split、balance及抽样图片解码
$PY vlm_ft/validate_high_vlm_v2_26tasks_dataset.py "$DATA"

# 5. 分层逐帧比较 JPEG 与 HDF5
$PY vlm_ft/audit_v2_26tasks_images.py "$DATA"
```

只有以下三个报告均为 `status: ok` 时才应启动训练：

```text
validation_report.json
image_materialization_report.json
stratified_image_audit.json
```

正式数据审计结果：14,800个 HDF5 全部重开检查；5,589,790张图像物化完成；3,322对分层 JPEG/HDF5 图像对照无错误。

## 5. 关键代码索引

| 文件 | 作用 |
|---|---|
| [prompt_contract_26.py](./prompt_contract_26.py) | 26任务描述、primitive 映射、训练/推理共用 prompt |
| [build_high_vlm_v2_26tasks_dataset.py](./build_high_vlm_v2_26tasks_dataset.py) | 从 HDF5 构造唯一训练样本和进度监督 |
| [make_balanced_train_v2_26tasks.py](./make_balanced_train_v2_26tasks.py) | 任务、primitive、关键帧和上下文平衡 |
| [materialize_v2_26tasks_images.py](./materialize_v2_26tasks_images.py) | 双相机 JPEG 物化与复用 |
| [validate_high_vlm_v2_26tasks_dataset.py](./validate_high_vlm_v2_26tasks_dataset.py) | 完整非图像审计与图片解码抽检 |
| [audit_v2_26tasks_images.py](./audit_v2_26tasks_images.py) | 分层 JPEG/HDF5 像素对照 |
| [training_components.py](./training_components.py) | V1 JSONL dataset、collator、LM loss |
| [train_three_task_lora.py](./train_three_task_lora.py) | V1 训练入口 |
| [high_vlm_v2_components.py](./high_vlm_v2_components.py) | 特殊 token、进度头、V2 collator/trainer/推理解码 |
| [train_high_vlm_v2.py](./train_high_vlm_v2.py) | V2 联合训练入口 |
| [eval_three_tasks.py](./eval_three_tasks.py) | GT 回放、闭环推理、逐调用结果保存 |
| [handoff_metrics.py](./handoff_metrics.py) | SCNA、切换延迟、提前切换与回退指标 |

## 6. 独立回放测评

`--action-source gt-replay` 在仿真器中回放成功演示动作，仅加载 VLM，不启动 VLA server。这样可以隔离低层控制失败，只测 VLM 的时序切分能力。

26任务数组作业：

```bash
cd /data/user/jwen341/openpi_rm

RUN_ROOT=/data/user/jwen341/openpi_rm/output/my_high_vlm_v2_26tasks_eval \
sbatch vlm_ft/slurm_eval_high_vlm_v2_26tasks_independent.sh
```

默认协议：

- 26任务，每任务10条独立 test seed，共260条轨迹；
- GT action replay；
- 每5个动作步同步调用一次 VLM；
- 最近5帧，最多8个预测关键帧；
- 保存所有原始输出、progress、prompt、实际输入图和视频；
- `trajectory-only` 模式按 HDF5 边界计算全26任务指标。

Slurm 脚本中的 partition、节点排除项、Python环境和路径是当前集群配置；迁移环境时需要修改这些字段。

### 6.1 切分指标

- `SCNA@0`：轨迹段完成后的第一次 VLM 调用必须预测 next primitive。
- `SCNA@2`：从边界后第0、1或2次调用开始，出现连续2次 next。
- `Switch Latency`：边界完成到稳定输出 next 的动作步/调用数延迟。
- `Premature Switch Rate`：边界前已经稳定切换到 next 的比例。
- `Post-switch Regression Rate`：正确稳定切换后又退回旧 primitive 的比例。
- `Progress MAE`：V2 预测进度和当前 HDF5 子轨迹归一化进度的绝对误差均值。

最终 close 没有 next，因此参与完整轨迹进度，但不进入 SCNA 边界分母。GT replay 的任务成功率按构造为100%，不能当作 VLM 性能指标。

### 6.2 输出结构

```text
RUN_ROOT/taskN/
├── run_config.json
├── aggregate.json
├── summary.tsv
├── boundary_metrics.jsonl
├── episode_summaries.jsonl
└── taskN/epXXX/
    ├── episode_summary.json
    ├── boundary_metrics.jsonl
    ├── vlm_predictions.jsonl
    ├── semantic_events.jsonl
    ├── sync_vlm.log
    ├── vlm_inputs/
    └── ...
```

`vlm_predictions.jsonl` 是误差分析的主要文件，包含：

- 完整 rendered prompt；
- `raw_output`、解析和规范化后的 primitive；
- `input_step`、`applied_step`、耗时；
- 关键帧输入/输出位置；
- `progress_prediction`、`progress_target` 和 absolute error。

## 7. 已知问题与开发约束

### 7.1 不要复用早期的 oracle-history 数据分布

早期 `three_tasks_strict_scna_v1` 中，后续 primitive 几乎总伴随 oracle history，而无历史样本基本只对应第一个 primitive。模型在闭环没有预测出关键帧时会退化为“无历史 → 第一个 primitive”。这也是早期 teacher-forced 验证 loss 很低、真实 SCNA 却很差的主要原因。

当前数据要求每个 primitive 都有无历史覆盖，并显式混合 none/partial/full history。

### 7.2 训练和推理必须使用同一 prompt

不要重新使用含有 “currently executing, or should execute now” 的歧义提示。任何 prompt、任务标签或图片排列的修改，都必须同时更新训练和推理，并重新运行数据验证。

### 7.3 最近5帧必须跨 HDF5 连续

不能在 primitive 边界补齐旧帧、复制边界帧或重置最近窗口。`R1...R5` 必须来自拼接后的任务级时间轴。

### 7.4 历史关键帧不能在边界清空

历史是整条长任务的记忆，不属于单个 primitive。只允许选择严格早于 R1 的帧，最多8个，禁止未来泄漏。

### 7.5 低验证 loss 不等于切分准确

checkpoint 可以按 `eval_loss` 保存，但最终选择必须参考独立 test seed 的 SCNA@0、逐 primitive 混淆和闭环关键帧行为。进度头也不能替代 primitive 切分指标。

### 7.6 运行时约20秒/调用通常不是输出过长

正常 H100 推理约1秒/调用，JSON通常16–17 token 后 EOS。历史上约20秒/调用并伴随 `SIGABRT` 或 `NODE_FAIL` 的情况来自节点/GPU运行时异常；先检查 GPU UUID、节点和 Slurm 退出码，不要直接归因于模型生成。

## 8. 快速检查

运行 V2 单元测试：

```bash
cd /data/user/jwen341/openpi_rm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=vlm_ft \
/data/user/hlei573/openpi_inference/.venv/bin/python -m unittest -v \
  vlm_ft/test_high_vlm_v2.py
```

检查指标实现：

```bash
cd /data/user/jwen341/openpi_rm/vlm_ft
python -m unittest -v test_semantic_metrics.py test_task_semantics.py
```

开始新实验前，建议依次确认：任务配置和 HDF5 stem 对齐、split 无泄漏、连续5帧、历史无未来帧、进度标签来自 R5、processor preflight 通过、两步 GPU smoke test 通过，最后再提交完整训练。
