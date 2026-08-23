#!/usr/bin/env python3
"""VLM/VLA evaluation with trajectory-SCNA and physical handoff metrics.

This runner evaluates the complete primitive sequences of RoboMemArena tasks
1, 17, 20 and 22.  It reuses the repository's reference VLM prompt, image
preprocessing, memory update, LIBERO environment and websocket VLA client, but
adds a primitive-aligned physical oracle and analysis-friendly traces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from semantic_metrics import (
    TASK_PRIMITIVES,
    PredictionRecord,
    PrimitiveNormalizer,
    aggregate_boundary_metrics,
    evaluate_boundary,
    summarize_records,
)
from task_semantics import OrderedSemanticTracker, build_completion_specs, physical_snapshot
from handoff_metrics import (
    VHWS_RULES,
    build_handoff_windows,
    evaluate_handoff_window,
    summarize_vhws,
    trajectory_boundaries,
)
from prompt_contract_26 import SYSTEM_PROMPT as STRICT_TRAJECTORY_SYSTEM_PROMPT
from prompt_contract_26 import build_runtime_messages, primitive_for_segment
from high_vlm_v2_components import (
    PROGRESS_HEAD_CONFIG_NAME,
    PROGRESS_TOKEN,
    FinalNormQueryCapture,
    get_progress_head,
    inject_progress_query,
    prepare_base_model_for_high_vlm_v2,
    progress_token_position,
    trim_generated_after_input,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/"
    "tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py"
)
DEFAULT_TASK_CONFIG = DEFAULT_REFERENCE.parent / "fullvlm_v2_26_memory_tasks.json"
SUPPORTED_TASKS = tuple(range(1, 27))
PHYSICAL_METRIC_TASKS = (1, 17, 20, 22)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _load_reference_runtime(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Reference evaluation runtime not found: {path}")
    os.environ.setdefault("PLANNER_MODE", "vlm")
    os.environ.setdefault("OPENPI_ROOT", str(REPO_ROOT))
    os.environ.setdefault(
        "TARGET_LIBERO_PATH",
        str(REPO_ROOT / "eval_robomem/robomemarena_official/evaluation_benchmark/libero_fork/libero"),
    )
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    compatibility_root = Path(os.environ.get("OUT_ROOT", "/tmp/vlm_ft_reference_import"))
    os.environ.setdefault("OUT_ROOT", str(compatibility_root))
    os.environ.setdefault("VIDEO_DIR", str(compatibility_root / "videos"))
    os.environ.setdefault("SUMMARY_JSON", str(compatibility_root / "summary.json"))
    os.environ.setdefault("SUMMARY_TSV", str(compatibility_root / "summary.tsv"))
    spec = importlib.util.spec_from_file_location("vlm_ft_reference_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json_arg(value: str) -> dict[str, Any]:
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint mapping must be a JSON object")
    return payload


def _resolve_checkpoints(args: argparse.Namespace) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    if args.vlm_checkpoints_json:
        for key, value in _load_json_arg(args.vlm_checkpoints_json).items():
            mapping[int(key)] = Path(str(value)).expanduser().resolve()
    if args.vlm_checkpoint:
        shared = Path(args.vlm_checkpoint).expanduser().resolve()
        for task_id in args.tasks:
            mapping.setdefault(int(task_id), shared)
    missing = [task_id for task_id in args.tasks if int(task_id) not in mapping]
    if missing:
        raise ValueError(
            "Missing VLM checkpoint(s) for task(s) "
            f"{missing}. Set --vlm-checkpoint or --vlm-checkpoints-json."
        )
    for task_id, path in mapping.items():
        if task_id in args.tasks and not path.is_dir():
            raise FileNotFoundError(f"VLM checkpoint for task{task_id} is not a directory: {path}")
    return {int(task_id): mapping[int(task_id)] for task_id in args.tasks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(SUPPORTED_TASKS))
    parser.add_argument("--vlm-checkpoint", default="", help="One checkpoint shared by all selected tasks.")
    parser.add_argument(
        "--vlm-checkpoints-json",
        default="",
        help='JSON file or object such as {"17":"/ckpt17","20":"/ckpt20","22":"/ckpt22"}.',
    )
    parser.add_argument("--processor-dir", default="", help="Optional canonical Qwen processor directory.")
    parser.add_argument(
        "--vlm-architecture",
        choices=("high_vlm_v1", "high_vlm_v2"),
        default="high_vlm_v1",
        help="high_vlm_v2 enables its input-only progress query and scalar progress head.",
    )
    parser.add_argument(
        "--vlm-base-model",
        default="",
        help=(
            "Optional shared base model for PEFT evaluation. When set, the resolved "
            "--vlm-checkpoint path is loaded as the LoRA adapter instead of as a full model."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-runtime", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--action-source", choices=("vla", "gt-replay"), default="vla")
    parser.add_argument(
        "--gt-data-root",
        type=Path,
        default=Path("/data/user/jwen341/dataset/robomemarena_fullvlm_v2_official_remote_20260711_raw"),
    )
    parser.add_argument("--gt-post-steps", type=int, default=30)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-model-type", default="qwen3_vl", choices=("qwen3_vl", "qwen2_5_vl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit independent-test seeds. Requires exactly one selected task and overrides --seed/--trials.",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--vlm-interval", type=int, default=5)
    parser.add_argument("--n-recent", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--d-merge", type=int, default=6)
    parser.add_argument("--vlm-queue-size", type=int, default=1)
    parser.add_argument("--async-vlm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-wrist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-keyframe-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vlm-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stable-env-steps", type=int, default=3)
    parser.add_argument("--stable-vlm-calls", type=int, default=2)
    parser.add_argument("--scna-tolerance-calls", type=int, default=2)
    parser.add_argument("--premature-tolerance-steps", type=int, default=1)
    parser.add_argument(
        "--trajectory-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drive stage transitions from original HDF5 segment endpoints. This supports all 26 tasks and "
            "reports trajectory SCNA/progress, but intentionally omits physical VHWS."
        ),
    )
    args = parser.parse_args()
    if not args.tasks or any(task_id not in SUPPORTED_TASKS for task_id in args.tasks):
        parser.error(f"--tasks must be a non-empty subset of {SUPPORTED_TASKS}")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks contains duplicates")
    if args.seeds is not None:
        if len(args.tasks) != 1:
            parser.error("--seeds requires exactly one task")
        if not args.seeds or len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must be a non-empty duplicate-free list")
        args.trials = len(args.seeds)
    if not args.trajectory_only and any(task_id not in PHYSICAL_METRIC_TASKS for task_id in args.tasks):
        parser.error(
            f"tasks outside {PHYSICAL_METRIC_TASKS} require --trajectory-only because physical VHWS rules "
            "have not been defined"
        )
    for name in (
        "trials",
        "max_steps",
        "replan_steps",
        "vlm_interval",
        "n_recent",
        "stable_env_steps",
        "stable_vlm_calls",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.scna_tolerance_calls < 0 or args.premature_tolerance_steps < 0:
        parser.error("metric tolerances must be non-negative")
    if args.gt_post_steps < 0:
        parser.error("--gt-post-steps must be non-negative")
    return args


def _load_gt_action_plan(data_root: Path, task_id: int, seed: int) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    task_dirs = sorted(data_root.glob(f"{task_id}_*_dataset"))
    if len(task_dirs) != 1:
        raise FileNotFoundError(
            f"Expected one task{task_id} dataset under {data_root}, found {len(task_dirs)}: {task_dirs}"
        )
    subtask_dir = task_dirs[0] / "subtask_data"
    records: list[tuple[int, Path]] = []
    for path in sorted(subtask_dir.glob(f"*_seed{seed}_task{task_id}.hdf5")):
        marker = f"_seed{seed}_task{task_id}"
        prefix = path.stem.removesuffix(marker)
        try:
            primitive_idx = int(prefix.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot parse primitive order from {path.name}") from exc
        records.append((primitive_idx, path))
    records.sort(key=lambda item: item[0])
    orders = [item[0] for item in records]
    if orders != list(range(len(records))):
        raise FileNotFoundError(
            f"Incomplete GT trajectory for task{task_id} seed{seed}: primitive orders={orders}"
        )

    actions: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for primitive_idx, path in records:
        marker = f"_seed{seed}_task{task_id}"
        prefix = path.stem.removesuffix(marker)
        stem = prefix.rsplit("_", 1)[0]
        label = primitive_for_segment(task_id, primitive_idx, stem)
        with h5py.File(path, "r") as handle:
            demo_keys = sorted(handle["data"].keys())
            if not demo_keys:
                raise ValueError(f"No demos in {path}")
            demo_key = demo_keys[0]
            primitive_actions = np.asarray(handle["data"][demo_key]["actions"], dtype=np.float32)
        start_step = len(actions)
        actions.extend(np.asarray(action, dtype=np.float32) for action in primitive_actions)
        manifest.append(
            {
                "primitive_index": primitive_idx,
                "primitive_stem": stem,
                "primitive_label": label,
                "path": str(path),
                "demo_key": demo_key,
                "action_count": int(len(primitive_actions)),
                "start_action_step": start_step,
                "end_action_step_exclusive": len(actions),
            }
        )
    return actions, manifest


@dataclass
class VLMRequest:
    request_index: int
    call_index: int
    input_step: int
    frames: list[tuple[np.ndarray, np.ndarray | None]]
    semantic_context: dict[str, Any]
    submitted_monotonic: float


@dataclass
class VLMResult:
    request: VLMRequest
    prompt: str
    event: dict[str, Any]
    error: str | None = None


def _make_recording_planner_class(ref: Any):
    class RecordingPlanner(ref.FullVlm26MemoryPlanner):
        """Reference planner variant that preserves the complete decoded output."""

        def __init__(
            self,
            *planner_args: Any,
            save_input_images: bool = True,
            high_vlm_v2_adapter_path: str = "",
            **planner_kwargs: Any,
        ) -> None:
            self.save_input_images = bool(save_input_images)
            self.high_vlm_v2_adapter_path = str(high_vlm_v2_adapter_path)
            self.high_vlm_v2_enabled = bool(self.high_vlm_v2_adapter_path)
            if self.high_vlm_v2_enabled:
                # The reference superclass loads the shared base.  The V2 head
                # must be attached before PEFT restores modules_to_save.
                planner_kwargs["lora_path"] = "none"
            super().__init__(*planner_args, **planner_kwargs)
            self.progress_token_id: int | None = None
            if self.high_vlm_v2_enabled:
                from peft import PeftModel

                adapter = Path(self.high_vlm_v2_adapter_path)
                config_path = adapter / PROGRESS_HEAD_CONFIG_NAME
                if not config_path.is_file():
                    raise FileNotFoundError(config_path)
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.progress_token_id = prepare_base_model_for_high_vlm_v2(
                    self.model,
                    self.processor.tokenizer,
                    config,
                )
                self.model = PeftModel.from_pretrained(self.model, adapter)
                self.model.eval()

        def _build_messages(
            self,
            memory_main_frames: list[Image.Image],
            memory_wrist_frames: list[Image.Image | None],
            context_main_frames: list[Image.Image],
            context_wrist_frames: list[Image.Image | None],
        ) -> list[dict[str, Any]]:
            """Use the exact training time/camera prompt contract."""
            current = self.step - 1
            recent_indices = list(range(current - len(context_main_frames) + 1, current + 1))
            return build_runtime_messages(
                task_id=int(self.task_info.task_id),
                history_indices=list(self.K_indices_abs),
                recent_indices=recent_indices,
                history_main=memory_main_frames,
                history_wrist=memory_wrist_frames,
                recent_main=context_main_frames,
                recent_wrist=context_wrist_frames,
            )

        def infer_record(self, request: VLMRequest) -> VLMResult:
            start_monotonic = time.monotonic()
            started_utc = _utc_now()
            try:
                prompt, event = self._infer_record_impl(request)
                event.update(
                    {
                        "started_utc": started_utc,
                        "completed_utc": _utc_now(),
                        "queue_delay_sec": start_monotonic - request.submitted_monotonic,
                        "inference_duration_sec": time.monotonic() - start_monotonic,
                    }
                )
                return VLMResult(request=request, prompt=prompt, event=event)
            except Exception as exc:
                return VLMResult(
                    request=request,
                    prompt="",
                    event={
                        "task_id": int(self.task_info.task_id),
                        "request_index": request.request_index,
                        "call_index": request.call_index,
                        "input_step": request.input_step,
                        "semantic_context": request.semantic_context,
                        "started_utc": started_utc,
                        "completed_utc": _utc_now(),
                        "queue_delay_sec": start_monotonic - request.submitted_monotonic,
                        "inference_duration_sec": time.monotonic() - start_monotonic,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )

        def _infer_record_impl(self, request: VLMRequest) -> tuple[str, dict[str, Any]]:
            step_idx = request.input_step
            context_frames_np = request.frames
            if not context_frames_np:
                raise ValueError("VLM request contains no recent frames")

            recent_start = step_idx - len(context_frames_np) + 1
            context_main_frames: list[Image.Image] = []
            context_wrist_frames: list[Image.Image | None] = []
            for offset, frame_pack in enumerate(context_frames_np):
                abs_idx = recent_start + offset
                main_np, wrist_np = frame_pack
                main_img = Image.fromarray(main_np.astype(np.uint8))
                wrist_img = (
                    Image.fromarray(wrist_np.astype(np.uint8))
                    if self.use_wrist and wrist_np is not None
                    else None
                )
                self.frame_store_main[abs_idx] = main_img
                self.frame_store_wrist[abs_idx] = wrist_img
                context_main_frames.append(main_img)
                context_wrist_frames.append(wrist_img)
            self.step = max(self.step, step_idx + 1)

            memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
            memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
            memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
            messages = self._build_messages(
                memory_main_frames,
                memory_wrist_frames,
                context_main_frames,
                context_wrist_frames,
            )
            images: list[Image.Image] = []
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    images.extend(
                        item["image"]
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "image"
                    )

            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(prompt_text, list):
                prompt_text = prompt_text[0]
            if self.high_vlm_v2_enabled:
                prompt_text, _ = inject_progress_query(prompt_text)
            inputs = self.processor(
                text=[prompt_text],
                images=images if images else None,
                return_tensors="pt",
                padding=False,
            )
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            progress_prediction: float | None = None
            if self.high_vlm_v2_enabled:
                if self.progress_token_id is None:
                    raise RuntimeError("high_vlm_v2 progress token was not initialized")
                query_position = progress_token_position(inputs["input_ids"][0], self.progress_token_id)
                if query_position != int(inputs["input_ids"].shape[1]) - 1:
                    raise ValueError("progress query is not the final generation-prompt token")
                positions = torch.tensor([query_position], device=inputs["input_ids"].device)
                with torch.inference_mode(), FinalNormQueryCapture(self.model, positions) as capture:
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        suppress_tokens=[self.progress_token_id],
                    )
                with torch.inference_mode():
                    progress_prediction = float(
                        torch.sigmoid(get_progress_head(self.model)(capture.require()).float())[0].item()
                    )
                trimmed = [
                    trim_generated_after_input(source, output)
                    for source, output in zip(inputs["input_ids"], generated, strict=True)
                ]
            else:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                trimmed = [output[len(source) :] for source, output in zip(inputs["input_ids"], generated)]
            generated_ids = [int(token) for token in trimmed[0].detach().cpu().tolist()]
            eos_value = self.processor.tokenizer.eos_token_id
            eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {
                int(token) for token in (eos_value or [])
            }
            terminated_by_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
            raw_decode_with_special_tokens = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            output_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed_primitive, keyframe_positions = ref._parse_output_no_mapping(
                output_text,
                max_pos=len(context_main_frames),
            )
            absolute_keyframes = [recent_start + position - 1 for position in keyframe_positions]

            if self.use_keyframe_memory:
                self.J_hist.append(absolute_keyframes)
                raw_indices = ref.build_visual_memory(
                    self.J_hist,
                    t=self.step,
                    N=len(context_main_frames),
                    d=self.d_merge,
                )
                self.K_indices_abs = [index for index in raw_indices if index < recent_start]
                self.K_main_frames = ref.get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
                self.K_wrist_frames = [self.frame_store_wrist.get(index) for index in self.K_indices_abs]
                if self.k_max > 0 and len(self.K_indices_abs) > self.k_max:
                    self.K_indices_abs = self.K_indices_abs[-self.k_max :]
                    self.K_main_frames = self.K_main_frames[-self.k_max :]
                    self.K_wrist_frames = self.K_wrist_frames[-self.k_max :]

            self._dump_new_keyframes()
            if parsed_primitive:
                self._current_subtask = parsed_primitive
            accepted_prompt = self._current_subtask
            image_bundle = None
            if self.run_dir is not None and self.save_input_images:
                image_bundle = self._save_vlm_input_bundle(
                    step_idx=step_idx,
                    memory_main_frames=memory_main_frames,
                    memory_wrist_frames=memory_wrist_frames,
                    memory_indices=memory_indices,
                    context_main_frames=context_main_frames,
                    context_wrist_frames=context_wrist_frames,
                    subtask=accepted_prompt,
                )

            trace = {
                "t": int(step_idx),
                "task_id": int(self.task_info.task_id),
                "subtask": accepted_prompt,
                "keyframe_positions": keyframe_positions,
                "J_abs": absolute_keyframes,
                "K_indices_abs": list(self.K_indices_abs),
                "out_text": output_text.strip(),
                "generated_token_count": len(generated_ids),
                "terminated_by_eos": terminated_by_eos,
                "generated_token_ids_tail": generated_ids[-16:],
                "progress_prediction": progress_prediction,
                "image": image_bundle,
                "request_index": request.request_index,
                "call_index": request.call_index,
            }
            self._append_trace(trace)
            if self.logger:
                self.logger.info(
                    "VLM call=%s input_t=%s primitive=%s progress=%s keyframes=%s",
                    request.call_index,
                    step_idx,
                    parsed_primitive,
                    progress_prediction,
                    keyframe_positions,
                )
            event = {
                "task_id": int(self.task_info.task_id),
                "request_index": request.request_index,
                "call_index": request.call_index,
                "input_step": int(step_idx),
                "recent_start_step": int(recent_start),
                "semantic_context": request.semantic_context,
                "rendered_prompt": prompt_text,
                "raw_output": output_text,
                "raw_output_with_special_tokens": raw_decode_with_special_tokens,
                "generated_token_count": len(generated_ids),
                "terminated_by_eos": terminated_by_eos,
                "generated_token_ids_tail": generated_ids[-16:],
                "progress_prediction": progress_prediction,
                "parsed_primitive": parsed_primitive,
                "accepted_controller_prompt": accepted_prompt,
                "keyframe_positions": keyframe_positions,
                "absolute_keyframe_positions": absolute_keyframes,
                "memory_indices_before_call": memory_indices,
                "memory_indices_after_call": list(self.K_indices_abs),
                "image_bundle": image_bundle,
            }
            return accepted_prompt, event

    return RecordingPlanner


def _clone_frames(frames: deque[tuple[np.ndarray, np.ndarray | None]]) -> list[tuple[np.ndarray, np.ndarray | None]]:
    return [(main.copy(), wrist.copy() if wrist is not None else None) for main, wrist in frames]


def _trajectory_progress_at_step(manifest: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    for row in manifest:
        start = int(row["start_action_step"])
        end = int(row["end_action_step_exclusive"])
        if start <= int(step) < end:
            length = int(row["action_count"])
            local_step = int(step) - start
            target = local_step / max(length - 1, 1)
            return {
                "progress_target": float(target),
                "progress_segment_index": int(row["primitive_index"]),
                "progress_segment_label": str(row["primitive_label"]),
                "progress_local_step": local_step,
                "progress_segment_length": length,
                "progress_near_start": local_step <= 4,
                "progress_near_end": end - int(step) <= 5,
            }
    return None


class OriginalTrajectoryTracker:
    """Stage tracker driven only by original HDF5 segment endpoints."""

    def __init__(self, manifest: list[dict[str, Any]]) -> None:
        if not manifest:
            raise ValueError("trajectory tracker requires a non-empty GT manifest")
        self.manifest = manifest
        self.primitives = [
            {
                "index": int(row["primitive_index"]),
                "label": str(row["primitive_label"]),
                "predicate_name": "original_hdf5_segment_endpoint",
                "activated_step": int(row["start_action_step"]),
                "activated_state_index": int(row["start_action_step"]),
                "completion_step": None,
                "completion_state_index": None,
            }
            for row in manifest
        ]
        self.current_index = 0

    @property
    def all_completed(self) -> bool:
        return self.current_index >= len(self.primitives)

    @property
    def current_label(self) -> str | None:
        return None if self.all_completed else str(self.primitives[self.current_index]["label"])

    @property
    def completed_count(self) -> int:
        return sum(row["completion_step"] is not None for row in self.primitives)

    def update(self, env: Any, state: dict[str, Any], step: int) -> dict[str, Any]:
        del env, state
        events: list[dict[str, Any]] = []
        evaluated_index = None if self.all_completed else self.current_index
        evaluated_label = self.current_label
        while not self.all_completed:
            source = self.manifest[self.current_index]
            boundary = int(source["end_action_step_exclusive"])
            if int(step) < boundary:
                break
            primitive = self.primitives[self.current_index]
            primitive["completion_step"] = boundary
            primitive["completion_state_index"] = boundary
            events.append(
                {
                    "event": "primitive_completed",
                    "step": boundary,
                    "state_index": boundary,
                    "primitive_index": self.current_index,
                    "primitive": primitive["label"],
                    "predicate_name": "original_hdf5_segment_endpoint",
                    "stable_steps": 1,
                }
            )
            self.current_index += 1
            if not self.all_completed:
                nxt = self.primitives[self.current_index]
                events.append(
                    {
                        "event": "primitive_activated",
                        "step": boundary,
                        "state_index": boundary,
                        "primitive_index": self.current_index,
                        "primitive": nxt["label"],
                    }
                )
            else:
                events.append({"event": "task_completed", "step": boundary, "state_index": boundary})
        return {
            "step": int(step),
            "evaluated_primitive_index": evaluated_index,
            "evaluated_primitive": evaluated_label,
            "predicate_name": "original_hdf5_segment_endpoint",
            "predicate_value": bool(events),
            "predicate_streak": 1 if events else 0,
            "completion_events": events,
            "current_primitive_index": None if self.all_completed else self.current_index,
            "current_primitive": self.current_label,
            "all_completed": self.all_completed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_type": "original_hdf5_trajectory",
            "current_index": self.current_index,
            "all_completed": self.all_completed,
            "completed_count": self.completed_count,
            "primitives": self.primitives,
        }


def _semantic_context(tracker: Any, step: int) -> dict[str, Any]:
    return {
        "input_step": int(step),
        "current_primitive_index": None if tracker.all_completed else tracker.current_index,
        "current_primitive": tracker.current_label,
        "completed_primitives": tracker.completed_count,
        "all_completed": tracker.all_completed,
    }


def _make_env(ref: Any, bddl_path: Path, task_id: int) -> Any:
    env_cls = ref.ec._get_env_class()
    retries = int(getattr(ref, "ENV_INIT_RETRIES", 3))
    retry_sleep = float(getattr(ref, "ENV_INIT_RETRY_SLEEP", 1.0))
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return env_cls(
                bddl_file_name=str(bddl_path),
                camera_heights=480,
                camera_widths=640,
                ignore_done=True,
                reward_shaping=True,
                control_freq=20,
                initialization_noise=None,
            )
        except Exception as exc:
            is_randomization_error = getattr(ref.stage_eval, "_is_randomization_error", lambda _: False)
            if not is_randomization_error(exc):
                raise
            last_error = exc
            logging.warning("task%s env init failed attempt %s/%s: %s", task_id, attempt, retries, exc)
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"task{task_id} env init failed after {retries} attempts: {last_error}")


def _run_episode(
    *,
    ref: Any,
    task_id: int,
    episode: int,
    seed: int,
    env: Any,
    client: Any | None,
    planner: Any,
    gt_actions: list[np.ndarray] | None,
    gt_manifest: list[dict[str, Any]] | None,
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    if args.trajectory_only and gt_manifest is None:
        raise ValueError("--trajectory-only requires --action-source gt-replay and a GT manifest")
    trajectory_labels = (
        [str(row["primitive_label"]) for row in gt_manifest]
        if gt_manifest is not None
        else None
    )
    normalizer = PrimitiveNormalizer(task_id, primitives=trajectory_labels)
    recent_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    requests: list[dict[str, Any]] = []
    results: list[VLMResult] = []
    semantic_events_path = run_dir / "semantic_events.jsonl"
    semantic_state_path = run_dir / "semantic_state.jsonl"
    for path in (semantic_events_path, semantic_state_path):
        path.write_text("", encoding="utf-8")

    obs = env.reset()
    for _ in range(args.num_steps_wait):
        obs, _, done, _ = env.step(ref.ec.LIBERO_DUMMY_ACTION)
        recent_frames.append(ref._extract_vlm_frame(env, obs, args, None))
        if done:
            logger.warning("Environment returned done during warmup")

    if args.trajectory_only:
        state = {"step_idx": 0, "last_obs": obs}
        tracker: Any = OriginalTrajectoryTracker(gt_manifest or [])
        initial_physical: dict[str, Any] = {}
    else:
        state = ref.stage_eval._build_initial_state(env)  # noqa: SLF001
        state["last_obs"] = obs
        specs = build_completion_specs(ref, task_id, stable_steps=args.stable_env_steps)
        tracker = OrderedSemanticTracker(specs, activation_step=0, state_index=int(state.get("step_idx", 0)))
        initial_physical = physical_snapshot(ref, env, state, task_id)
    _append_jsonl(
        semantic_events_path,
        {
            "event": "primitive_activated",
            "step": 0,
            "state_index": int(state.get("step_idx", 0)),
            "primitive_index": 0,
            "primitive": tracker.current_label,
        },
    )
    _append_jsonl(
        semantic_state_path,
        {
            "step": 0,
            "tracker": tracker.to_dict(),
            "physical": initial_physical,
        },
    )

    request_queue: queue.Queue[VLMRequest | None] | None = None
    result_queue: queue.Queue[VLMResult] | None = None
    worker: threading.Thread | None = None
    stop_worker = threading.Event()
    if args.async_vlm:
        request_queue = queue.Queue(maxsize=max(1, args.vlm_queue_size))
        result_queue = queue.Queue()

        def worker_loop() -> None:
            assert request_queue is not None and result_queue is not None
            while not stop_worker.is_set():
                try:
                    request = request_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if request is None:
                    break
                result_queue.put(planner.infer_record(request))

        worker = threading.Thread(target=worker_loop, name=f"vlm-ft-task{task_id}-ep{episode}", daemon=True)
        worker.start()

    next_query_step = 0
    request_index = 0
    call_index = 0
    current_prompt = planner.default_subtask_prompt
    last_vla_prompt: str | None = None
    prompt_switches = 0
    vla_chunks = 0
    env_step = 0
    failure_reason: str | None = None
    gt_action_index = 0
    gt_post_steps_used = 0
    if gt_manifest is not None:
        _write_json(run_dir / "gt_replay_manifest.json", gt_manifest)

    def submit_request(step: int) -> None:
        nonlocal request_index, call_index
        request = VLMRequest(
            request_index=request_index,
            call_index=call_index,
            input_step=int(step),
            frames=_clone_frames(recent_frames),
            semantic_context=_semantic_context(tracker, step),
            submitted_monotonic=time.monotonic(),
        )
        request_row = {
            "request_index": request_index,
            "call_index": call_index,
            "input_step": int(step),
            "semantic_context": request.semantic_context,
            "submitted_utc": _utc_now(),
            "status": "submitted",
        }
        requests.append(request_row)
        request_index += 1
        call_index += 1
        if not args.async_vlm:
            result = planner.infer_record(request)
            result.event["applied_step"] = int(step)
            result.event["applied_after_episode"] = False
            results.append(result)
            return

        assert request_queue is not None
        try:
            request_queue.put_nowait(request)
        except queue.Full:
            try:
                dropped = request_queue.get_nowait()
            except queue.Empty:
                dropped = None
            if dropped is not None:
                for row in requests:
                    if row["request_index"] == dropped.request_index:
                        row["status"] = "dropped_before_inference"
                        break
            try:
                request_queue.put_nowait(request)
            except queue.Full:
                request_row["status"] = "dropped_before_inference"

    def apply_ready(step: int, *, after_episode: bool = False) -> None:
        nonlocal current_prompt, prompt_switches, failure_reason
        if not args.async_vlm or result_queue is None:
            candidates = [item for item in results if "applied_step" not in item.event]
        else:
            candidates = []
            while True:
                try:
                    candidates.append(result_queue.get_nowait())
                except queue.Empty:
                    break
        for result in candidates:
            result.event["applied_step"] = int(step)
            result.event["applied_after_episode"] = bool(after_episode)
            # apply_ready is only used for queued async results; synchronous
            # results are appended directly by submit_request.
            if args.async_vlm:
                results.append(result)
            for row in requests:
                if row["request_index"] == result.request.request_index:
                    row["status"] = "inference_error" if result.error else "completed"
                    break
            if result.error:
                failure_reason = f"vlm_inference_error:{result.error}"
                logger.error("VLM inference failed: %s", result.error)
                continue
            if result.prompt and result.prompt != current_prompt:
                prompt_switches += 1
                current_prompt = result.prompt
                logger.info("[t=%s] applied VLM call=%s prompt=%s", step, result.request.call_index, current_prompt)

    try:
        while env_step < args.max_steps and not tracker.all_completed and failure_reason is None:
            apply_ready(env_step)
            if env_step >= next_query_step and len(recent_frames) >= args.n_recent:
                submit_request(env_step)
                if not args.async_vlm:
                    # Synchronous result is already present and is applied now.
                    sync_result = results[-1]
                    for row in requests:
                        if row["request_index"] == sync_result.request.request_index:
                            row["status"] = "inference_error" if sync_result.error else "completed"
                    if sync_result.error:
                        failure_reason = f"vlm_inference_error:{sync_result.error}"
                    elif sync_result.prompt and sync_result.prompt != current_prompt:
                        prompt_switches += 1
                        current_prompt = sync_result.prompt
                next_query_step += args.vlm_interval
                if failure_reason is not None:
                    break

            if args.async_vlm:
                apply_ready(env_step)

            prompt_for_vla = current_prompt or planner.default_subtask_prompt
            if last_vla_prompt is not None and prompt_for_vla != last_vla_prompt:
                logger.info("[t=%s] VLA prompt switch: %s -> %s", env_step, last_vla_prompt, prompt_for_vla)
            last_vla_prompt = prompt_for_vla
            if args.action_source == "vla":
                if client is None:
                    raise RuntimeError("VLA action source selected without a websocket client")
                element = ref.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                output = client.infer(element)
                actions = np.asarray(output["actions"])
                vla_chunks += 1
            elif gt_actions is not None and gt_action_index < len(gt_actions):
                actions = np.asarray(gt_actions[gt_action_index : gt_action_index + args.replan_steps])
            elif gt_post_steps_used < args.gt_post_steps:
                actions = np.asarray([ref.ec.LIBERO_DUMMY_ACTION])
                gt_post_steps_used += 1
            else:
                failure_reason = "gt_replay_actions_exhausted"
                break

            action_limit = min(args.replan_steps, len(actions), args.max_steps - env_step)
            if not args.async_vlm:
                # Reach every requested VLM tick exactly even when replan_steps
                # is larger than vlm_interval.
                action_limit = min(action_limit, max(1, next_query_step - env_step))

            for action in actions[:action_limit]:
                element_step = ref.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                replay.append(np.asarray(element_step["observation/image"]))
                wrist_image = element_step.get("observation/wrist_image")
                if wrist_image is not None:
                    replay_wrist.append(np.asarray(wrist_image))

                obs, _, done, _ = env.step(np.asarray(action).tolist())
                if args.action_source == "gt-replay" and gt_action_index < len(gt_actions or []):
                    gt_action_index += 1
                env_step += 1
                if args.trajectory_only:
                    state["step_idx"] = env_step
                    state["last_obs"] = obs
                else:
                    ref.stage_eval._update_state(obs, state)  # noqa: SLF001
                recent_frames.append(ref._extract_vlm_frame(env, obs, args, None))
                tracker_row = tracker.update(env, state, env_step)
                for event in tracker_row["completion_events"]:
                    _append_jsonl(semantic_events_path, event)
                    logger.info("semantic event: %s", event)
                _append_jsonl(
                    semantic_state_path,
                    {
                        **tracker_row,
                        "physical": (
                            {} if args.trajectory_only else physical_snapshot(ref, env, state, task_id)
                        ),
                        "controller_prompt": prompt_for_vla,
                    },
                )

                if args.async_vlm and env_step >= next_query_step and not tracker.all_completed:
                    submit_request(env_step)
                    next_query_step += args.vlm_interval
                if done:
                    failure_reason = "environment_done"
                    break
                if tracker.all_completed or failure_reason is not None:
                    break

            if action_limit == 0:
                failure_reason = "vla_returned_no_actions"
    except Exception as exc:
        failure_reason = f"episode_exception:{type(exc).__name__}:{exc}"
        logger.exception("Episode failed")
    finally:
        if args.async_vlm and request_queue is not None:
            stop_worker.set()
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                pass
            if worker is not None:
                worker.join(timeout=120.0)
                if worker.is_alive():
                    logger.error("VLM worker did not terminate within 120 seconds")
                    failure_reason = failure_reason or "vlm_worker_shutdown_timeout"
            apply_ready(env_step, after_episode=True)
            for row in requests:
                if row["status"] == "submitted":
                    row["status"] = "not_run_before_episode_end"

    if not tracker.all_completed and failure_reason is None and env_step >= args.max_steps:
        failure_reason = "max_steps"

    normalized_events: list[dict[str, Any]] = []
    prediction_records: list[PredictionRecord] = []
    for result in sorted(results, key=lambda item: item.request.call_index):
        event = dict(result.event)
        if result.error:
            canonical, method = "__unknown__", "inference_error"
            parsed = ""
        else:
            parsed = str(event.get("parsed_primitive", ""))
            canonical, method = normalizer.normalize(parsed)
        event["episode"] = int(episode)
        event["seed"] = int(seed)
        event["normalized_primitive"] = canonical
        event["normalization_method"] = method
        if gt_manifest is not None:
            progress_truth = _trajectory_progress_at_step(gt_manifest, int(event["input_step"]))
            if progress_truth is not None:
                event.update(progress_truth)
                prediction = event.get("progress_prediction")
                if prediction is not None:
                    event["progress_absolute_error"] = abs(
                        float(prediction) - float(progress_truth["progress_target"])
                    )
        normalized_events.append(event)
        prediction_records.append(
            PredictionRecord(
                call_index=int(event["call_index"]),
                input_step=int(event["input_step"]),
                parsed_primitive=parsed,
                normalized_primitive=canonical,
                applied_step=event.get("applied_step"),
            )
        )

    predictions_path = run_dir / "vlm_predictions.jsonl"
    requests_path = run_dir / "vlm_requests.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    requests_path.write_text("", encoding="utf-8")
    for row in normalized_events:
        _append_jsonl(predictions_path, row)
    for row in requests:
        _append_jsonl(requests_path, row)

    boundary_metrics: list[dict[str, Any]] = []
    boundaries = [] if gt_manifest is None else trajectory_boundaries(
        task_id, episode, seed, gt_manifest, env_step, labels=trajectory_labels
    )
    for boundary in boundaries:
        boundary_metrics.append(
            evaluate_boundary(
                boundary,
                prediction_records,
                stable_calls=args.stable_vlm_calls,
                scna_tolerance_calls=args.scna_tolerance_calls,
                premature_tolerance_steps=args.premature_tolerance_steps,
            )
        )

    boundary_path = run_dir / "boundary_metrics.jsonl"
    boundary_path.write_text("", encoding="utf-8")
    for row in boundary_metrics:
        _append_jsonl(boundary_path, row)

    semantic_states = [
        json.loads(line)
        for line in semantic_state_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    handoff_metrics = [] if args.trajectory_only else [
        evaluate_handoff_window(window, prediction_records, stable_calls=args.stable_vlm_calls)
        for window in build_handoff_windows(task_id, episode, seed, tracker.to_dict(), semantic_states)
    ]
    handoff_path = run_dir / "handoff_metrics.jsonl"
    handoff_path.write_text("", encoding="utf-8")
    for row in handoff_metrics:
        _append_jsonl(handoff_path, row)

    segmentation_summary = summarize_records(boundary_metrics)
    progress_errors = [
        float(row["progress_absolute_error"])
        for row in normalized_events
        if row.get("progress_absolute_error") is not None
    ]
    near_start_errors = [
        float(row["progress_absolute_error"])
        for row in normalized_events
        if row.get("progress_absolute_error") is not None and row.get("progress_near_start")
    ]
    near_end_errors = [
        float(row["progress_absolute_error"])
        for row in normalized_events
        if row.get("progress_absolute_error") is not None and row.get("progress_near_end")
    ]
    progress_summary = {
        "num_predictions": len(progress_errors),
        "mae": float(np.mean(progress_errors)) if progress_errors else None,
        "near_start_mae": float(np.mean(near_start_errors)) if near_start_errors else None,
        "near_start_count": len(near_start_errors),
        "near_end_mae": float(np.mean(near_end_errors)) if near_end_errors else None,
        "near_end_count": len(near_end_errors),
    }
    summary = {
        "task_id": task_id,
        "episode": episode,
        "seed": seed,
        "task_success": tracker.all_completed,
        "completed_primitives": tracker.completed_count,
        "total_primitives": len(tracker.primitives),
        "semantic_progress": tracker.completed_count / len(tracker.primitives),
        "episode_end_step": env_step,
        "failure_reason": failure_reason,
        "action_source": args.action_source,
        "num_gt_actions": 0 if gt_actions is None else len(gt_actions),
        "num_gt_actions_executed": gt_action_index,
        "final_controller_prompt": current_prompt,
        "num_vla_chunks": vla_chunks,
        "num_prompt_switches": prompt_switches,
        "num_vlm_requests": len(requests),
        "num_vlm_calls": len(results),
        "num_dropped_vlm_requests": sum(row["status"] == "dropped_before_inference" for row in requests),
        "num_unfinished_vlm_requests": sum(row["status"] == "not_run_before_episode_end" for row in requests),
        "semantic_tracker": tracker.to_dict(),
        "segmentation": segmentation_summary,
        "progress": progress_summary,
        "vhws": summarize_vhws(handoff_metrics),
        "task_success_definition": (
            "original_hdf5_trajectory_replay_completed"
            if args.trajectory_only
            else "physical_semantic_tracker_completed"
        ),
    }
    _write_json(run_dir / "episode_summary.json", summary)
    return summary, boundary_metrics, handoff_metrics, replay, replay_wrist


def _summary_tsv_header() -> str:
    return (
        "task_id\tepisode\tseed\ttask_success\tsemantic_progress\tepisode_end_step\tvlm_calls\t"
        "scna0\tscna2\tstable_switch_coverage\tswitch_latency_steps_mean\t"
        "switch_latency_steps_median\tpremature_switch_rate\tpost_switch_regression_rate\t"
        "progress_mae\tfailure_reason\n"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoints = _resolve_checkpoints(args)
    base_model = Path(args.vlm_base_model).expanduser().resolve() if args.vlm_base_model else None
    if base_model is not None and not base_model.is_dir():
        raise FileNotFoundError(f"VLM base model is not a directory: {base_model}")
    if args.vlm_architecture == "high_vlm_v2" and base_model is None:
        raise ValueError("high_vlm_v2 evaluation requires --vlm-base-model plus a PEFT adapter checkpoint")
    out_root = Path(args.output_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Output directory must be empty or absent: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    video_root = out_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    global_boundary_path = out_root / "boundary_metrics.jsonl"
    global_episode_path = out_root / "episode_summaries.jsonl"
    global_handoff_path = out_root / "handoff_metrics.jsonl"
    summary_tsv = out_root / "summary.tsv"
    for path in (global_boundary_path, global_episode_path, global_handoff_path):
        path.write_text("", encoding="utf-8")
    summary_tsv.write_text(_summary_tsv_header(), encoding="utf-8")

    run_config = {
        "created_utc": _utc_now(),
        "command": sys.argv,
        "arguments": vars(args),
        "vlm_checkpoints": {str(key): str(value) for key, value in checkpoints.items()},
        "vlm_base_model": None if base_model is None else str(base_model),
        "vlm_architecture": args.vlm_architecture,
        "checkpoint_loading": "base_plus_lora" if base_model is not None else "full_model",
        "metric_contract": {
            "action_source": args.action_source,
            "prediction_source": "raw parsed VLM output normalized by an explicit alias table",
            "scna_ground_truth": "original HDF5 segment end_action_step_exclusive",
            "semantic_ground_truth": "used only to construct physical VHWS windows and task progress",
            "scna0": "first VLM call at/after the original trajectory boundary predicts next",
            "scna2": (
                f"a {args.stable_vlm_calls}-call stable next run starts at k0..k{args.scna_tolerance_calls}"
            ),
            "switch_latency": "start of first stable next run minus original trajectory boundary",
            "premature_switch": "stable next before the original trajectory boundary",
            "post_switch_regression": "stable old run after stable next and before the following trajectory boundary",
            "vhws": "at least one next / one stable-next run is observed inside the task-specific physical handoff window",
            "vhws_rules": {str(task): rules for task, rules in VHWS_RULES.items() if task in args.tasks},
            "final_primitive": "tracked for task success but excluded from SCNA because it has no next primitive",
        },
    }
    _write_json(out_root / "run_config.json", run_config)

    os.environ["OUT_ROOT"] = str(out_root)
    os.environ["VIDEO_DIR"] = str(video_root)
    os.environ["SUMMARY_JSON"] = str(out_root / "reference_compat_summary.json")
    os.environ["SUMMARY_TSV"] = str(out_root / "reference_compat_summary.tsv")
    os.environ["TASKS_JSON"] = json.dumps(args.tasks)
    ref = _load_reference_runtime(args.reference_runtime.resolve())
    ref.patch_env_resolution()
    # _extract_vlm_frame consumes the reference Args preprocessing fields.  We
    # keep the CLI focused, then explicitly select the training-matched profile.
    args.vlm_input_profile = "fullvlm_256"
    args.vlm_match_vla_preprocess = False
    args.vlm_use_openpi_camera_pose = False
    args.vlm_render_height = 720
    args.vlm_render_width = 1280
    args.vlm_match_training_jpeg_roundtrip = False
    args.vlm_training_jpeg_quality = 30
    args.vlm_use_wrist = args.use_wrist
    args.vlm_wrist_required = False
    ref._apply_vlm_input_profile(args)
    task_infos = ref.load_task_infos(args.task_config.resolve())
    RecordingPlanner = _make_recording_planner_class(ref)
    ref._seed_everywhere(args.seed)
    client = None
    if args.action_source == "vla":
        client = ref.StableWebsocketClientPolicy(
            args.host,
            args.port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=30.0,
        )
    all_boundaries: list[dict[str, Any]] = []
    all_handoffs: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    try:
        for task_id in args.tasks:
            task_info = task_infos[task_id]
            checkpoint = checkpoints[task_id]
            logging.info("Loading task%s VLM checkpoint: %s", task_id, checkpoint)
            planner_kwargs = {
                "base_model_dir": str(base_model if base_model is not None else checkpoint),
                "lora_path": str(checkpoint) if base_model is not None else "none",
                "instruction": "",
                "system_prompt": STRICT_TRAJECTORY_SYSTEM_PROMPT,
                "prompt_profile": "task1_kf5",
                "n_recent": args.n_recent,
                "d_merge": args.d_merge,
                "k_max": args.k_max,
                "use_keyframe_memory": args.use_keyframe_memory,
                "max_new_tokens": args.max_new_tokens,
                "device": args.vlm_device,
                "logger": None,
                "vlm_model_type": args.vlm_model_type,
                "enable_thinking": False,
                "crop_right_half": False,
                "use_wrist": args.use_wrist,
                "task_info": task_info,
                "save_input_images": args.save_vlm_images,
            }
            if args.vlm_architecture == "high_vlm_v2":
                planner_kwargs["high_vlm_v2_adapter_path"] = str(checkpoint)
            if args.processor_dir:
                planner_kwargs["processor_model_dir"] = args.processor_dir
            planner = RecordingPlanner(**planner_kwargs)
            bddl_path = ref.ec._resolve_bddl_path(task_id)
            env = _make_env(ref, bddl_path, task_id)
            try:
                episode_seeds = args.seeds or [args.seed + episode for episode in range(args.trials)]
                for episode, seed in enumerate(episode_seeds):
                    ref._seed_everywhere(seed)
                    try:
                        env.seed(seed)
                    except AttributeError:
                        pass
                    run_dir = out_root / f"task{task_id}" / f"ep{episode:03d}"
                    logger = ref.make_episode_logger(run_dir)
                    logger.info(
                        "task=%s episode=%s seed=%s checkpoint=%s bddl=%s",
                        task_id,
                        episode,
                        seed,
                        checkpoint,
                        bddl_path,
                    )
                    planner.reset_episode(instruction="", run_dir=run_dir, logger=logger)
                    gt_actions = None
                    gt_manifest = None
                    if args.action_source == "gt-replay":
                        gt_actions, gt_manifest = _load_gt_action_plan(
                            args.gt_data_root.resolve(), task_id, seed
                        )
                    summary, boundaries, handoffs, replay, replay_wrist = _run_episode(
                        ref=ref,
                        task_id=task_id,
                        episode=episode,
                        seed=seed,
                        env=env,
                        client=client,
                        planner=planner,
                        gt_actions=gt_actions,
                        gt_manifest=gt_manifest,
                        args=args,
                        run_dir=run_dir,
                        logger=logger,
                    )
                    all_episodes.append(summary)
                    all_boundaries.extend(boundaries)
                    all_handoffs.extend(handoffs)
                    _append_jsonl(global_episode_path, summary)
                    for row in boundaries:
                        _append_jsonl(global_boundary_path, row)
                    for row in handoffs:
                        _append_jsonl(global_handoff_path, row)
                    segment = summary["segmentation"]
                    with summary_tsv.open("a", encoding="utf-8") as stream:
                        stream.write(
                            "\t".join(
                                _fmt(value)
                                for value in (
                                    task_id,
                                    episode,
                                    seed,
                                    int(summary["task_success"]),
                                    summary["semantic_progress"],
                                    summary["episode_end_step"],
                                    summary["num_vlm_calls"],
                                    segment["scna0"],
                                    segment["scna2"],
                                    segment["stable_switch_coverage"],
                                    segment["switch_latency_steps_mean"],
                                    segment["switch_latency_steps_median"],
                                    segment["premature_switch_rate"],
                                    segment["post_switch_regression_rate"],
                                    summary["progress"]["mae"],
                                    summary["failure_reason"],
                                )
                            )
                            + "\n"
                        )
                    if args.save_video and replay:
                        ref._write_video(video_root / f"task{task_id}_ep{episode:03d}_seed{seed}_main.mp4", replay, fps=10)
                    if args.save_video and replay_wrist:
                        ref._write_video(
                            video_root / f"task{task_id}_ep{episode:03d}_seed{seed}_wrist.mp4",
                            replay_wrist,
                            fps=10,
                        )
                    logging.info(
                        "task%s ep%s success=%s progress=%.3f SCNA0=%s SCNA2=%s latency=%s",
                        task_id,
                        episode,
                        summary["task_success"],
                        summary["semantic_progress"],
                        segment["scna0"],
                        segment["scna2"],
                        segment["switch_latency_steps_mean"],
                    )
            finally:
                env.close()
                planner.close()
                del planner
                try:
                    __import__("torch").cuda.empty_cache()
                except Exception:
                    pass
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    aggregate = aggregate_boundary_metrics(all_boundaries)
    aggregate["vhws"] = summarize_vhws(all_handoffs)
    aggregate["vhws_per_task"] = {
        str(task_id): summarize_vhws([row for row in all_handoffs if row["task_id"] == task_id])
        for task_id in args.tasks
    }
    per_task_episode: dict[str, Any] = {}
    for task_id in args.tasks:
        rows = [row for row in all_episodes if row["task_id"] == task_id]
        progress_count = sum(int(row["progress"]["num_predictions"]) for row in rows)
        progress_weighted_error = sum(
            float(row["progress"]["mae"]) * int(row["progress"]["num_predictions"])
            for row in rows
            if row["progress"]["mae"] is not None
        )
        per_task_episode[str(task_id)] = {
            "episodes": len(rows),
            "task_success_rate": sum(bool(row["task_success"]) for row in rows) / max(1, len(rows)),
            "mean_semantic_progress": sum(float(row["semantic_progress"]) for row in rows) / max(1, len(rows)),
            "mean_episode_end_step": sum(int(row["episode_end_step"]) for row in rows) / max(1, len(rows)),
            "progress_predictions": progress_count,
            "progress_mae": progress_weighted_error / progress_count if progress_count else None,
        }
    aggregate["episode_outcomes"] = {
        "per_task": per_task_episode,
        "macro_task_success_rate": sum(row["task_success_rate"] for row in per_task_episode.values())
        / max(1, len(per_task_episode)),
        "macro_semantic_progress": sum(row["mean_semantic_progress"] for row in per_task_episode.values())
        / max(1, len(per_task_episode)),
    }
    total_progress_predictions = sum(row["progress_predictions"] for row in per_task_episode.values())
    aggregate["progress"] = {
        "num_predictions": total_progress_predictions,
        "micro_mae": (
            sum(
                float(row["progress_mae"]) * int(row["progress_predictions"])
                for row in per_task_episode.values()
                if row["progress_mae"] is not None
            )
            / total_progress_predictions
            if total_progress_predictions else None
        ),
        "macro_task_mae": (
            sum(float(row["progress_mae"]) for row in per_task_episode.values() if row["progress_mae"] is not None)
            / max(1, sum(row["progress_mae"] is not None for row in per_task_episode.values()))
        ),
    }
    _write_json(out_root / "aggregate.json", aggregate)
    logging.info("Evaluation complete: %s", out_root / "aggregate.json")


if __name__ == "__main__":
    main()
