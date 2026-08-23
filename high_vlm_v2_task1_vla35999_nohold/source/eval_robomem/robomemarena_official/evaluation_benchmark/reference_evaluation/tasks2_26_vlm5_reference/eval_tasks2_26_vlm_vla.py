from __future__ import annotations

import json
import logging
import os
import queue
import argparse
import inspect
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import tqdm
import torch
from PIL import Image


REFERENCE_DIR = Path(__file__).resolve().parent
EVAL_BENCHMARK_DIR = REFERENCE_DIR.parents[1]
REPO_ROOT = EVAL_BENCHMARK_DIR.parent
RUNTIME_DIR = EVAL_BENCHMARK_DIR / "openpi_minimal_runtime"
DEFAULT_OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi_minimal"
ROOT = Path(os.environ.get("OPENPI_ROOT", str(DEFAULT_OPENPI_ROOT))).resolve()
_default_inference_root = REPO_ROOT.parent / "openpi_inference"
if not _default_inference_root.exists():
    _default_inference_root = REPO_ROOT / "openpi_inference"
INFERENCE_ROOT = Path(
    os.environ.get("OPENPI_INFERENCE_ROOT", str(_default_inference_root))
).resolve()
OPENPI_CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
OPENPI_SRC = ROOT / "packages" / "openpi" / "src"
LIBERO_PATH_ENV = os.environ.get("TARGET_LIBERO_PATH", "").strip()
if not LIBERO_PATH_ENV:
    _fallback_libero = EVAL_BENCHMARK_DIR / "libero_fork" / "libero"
    if _fallback_libero.exists():
        LIBERO_PATH_ENV = str(_fallback_libero)
LIBERO_PATHS: list[Path] = []
if LIBERO_PATH_ENV:
    _libero_path = Path(LIBERO_PATH_ENV)
    LIBERO_PATHS.extend([_libero_path, _libero_path.parent])

module_paths = [
    str(RUNTIME_DIR),
    str(OPENPI_CLIENT_SRC),
    str(OPENPI_SRC),
]
for _lib_path in LIBERO_PATHS:
    module_paths.append(str(_lib_path))

for p in module_paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import eval_common as ec
import retry_tasks2_26_stage_from_anygrasp as stage_eval
from keyframe_selection import build_visual_memory, get_frames_from_indices
from robocerebra_adapter import obs_to_pi_element

ENV_INIT_RETRIES = int(os.environ.get("ENV_INIT_RETRIES", str(stage_eval.ENV_INIT_RETRIES)))
ENV_INIT_RETRY_SLEEP = float(os.environ.get("ENV_INIT_RETRY_SLEEP", str(stage_eval.ENV_INIT_RETRY_SLEEP)))

if os.environ.get("PLANNER_MODE", "vlm").strip().lower() == "gt_subtask":
    @dataclass
    class BaseArgs:
        host: str = "127.0.0.1"
        port: int = 8026
        base_model_dir: str = ""
        lora_path: str = "none"
        vlm_device: str = "cuda:1"
        resize_size: int = 256
        replan_steps: int = 5
        num_steps_wait: int = 10
        max_steps: int = 2000
        seed: int = 42
        num_trials_per_task: int = 1
        vlm_input_profile: str = "fullvlm_256"
        vlm_match_training_jpeg_roundtrip: bool = False
        vlm_training_jpeg_quality: int = 30
        async_vlm: bool = False
        vlm_interval: int = 5
        vlm_queue_size: int = 1
        n_recent: int = 5
        k_max: int = 0
        d_merge: int = 6
        vlm_use_wrist: bool = True
        vlm_use_keyframe_memory: bool = True
        vlm_model_type: str = "qwen3_vl"
        vlm_match_vla_preprocess: bool = False
        vlm_use_openpi_camera_pose: bool = False
        vlm_render_height: int = 720
        vlm_render_width: int = 1280
        vlm_match_fullvlm_source_square: bool = True
        vlm_source_square_size: int = 256
        vlm_resize_for_training: bool = True
        vlm_train_width: int = 256
        vlm_train_height: int = 256
        vlm_wrist_required: bool = False

    class StableWebsocketClientPolicy(ec._websocket_client_policy.WebsocketClientPolicy):
        def __init__(
            self,
            host: str = "127.0.0.1",
            port: int | None = None,
            api_key: str | None = None,
            *,
            ping_interval: float | None = None,
            ping_timeout: float | None = None,
            close_timeout: float = 30.0,
        ) -> None:
            _ = (ping_interval, ping_timeout, close_timeout)
            super().__init__(host=host, port=port, api_key=api_key)

    class SyncLoRAPlanner:
        pass

    def _seed_everywhere(seed: int) -> None:
        np.random.seed(seed)

    def _apply_vlm_input_profile(args: BaseArgs) -> None:
        profile = (args.vlm_input_profile or "").strip().lower()
        if profile == "custom":
            return
        if profile == "fullvlm_256":
            args.vlm_match_fullvlm_source_square = True
            args.vlm_source_square_size = 256
            args.vlm_resize_for_training = True
            args.vlm_train_width = 256
            args.vlm_train_height = 256
            return
        if profile == "task1_768":
            args.vlm_match_fullvlm_source_square = False
            args.vlm_resize_for_training = True
            args.vlm_train_width = 768
            args.vlm_train_height = 432
            return
        if profile == "task1_1080":
            args.vlm_match_fullvlm_source_square = False
            args.vlm_resize_for_training = True
            args.vlm_train_width = 1080
            args.vlm_train_height = 1080
            return
        raise ValueError(f"Unknown vlm_input_profile={args.vlm_input_profile!r}")

    def _extract_vlm_frame(*args: Any, **kwargs: Any):
        raise RuntimeError("_extract_vlm_frame is only available in planner-mode=vlm")

    def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
        if not frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        norm_frames = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            norm_frames.append(arr)
        try:
            imageio.mimwrite(path, norm_frames, fps=fps, codec="libx264")
            return
        except Exception as exc:
            logging.warning("imageio video write failed, falling back to cv2: %s", exc)
        import cv2

        h, w = norm_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        try:
            for frame in norm_frames:
                cur = frame
                if cur.shape[:2] != (h, w):
                    cur = cv2.resize(cur, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(cur, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def make_episode_logger(run_dir: Path) -> logging.Logger:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"gt_subtask_{run_dir.parent.name}_{run_dir.name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(run_dir / "gt_subtask.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
        return logger
else:
    from eval_task1_qwen3_async_openpi_inference_vla_cam import (
        Args as BaseArgs,
        StableWebsocketClientPolicy,
        SyncLoRAPlanner,
        _apply_vlm_input_profile,
        _extract_vlm_frame,
        _seed_everywhere,
        _write_video,
        make_episode_logger,
    )


SYSTEM_PROMPT_MEMORY_DEMO = """You are an embodied-memory robot VLM planner.

You will observe two kinds of visual evidence from the same long-horizon execution:
1. Historical keyframes: moments before the current step, used to remember important past states.
2. A recent 5-frame dual-camera window ending at the current frame, used to infer the current primitive.
Temporal order: historical keyframes are ordered from earliest to latest; the recent 5-frame window is also ordered from earliest to latest, and the last timestep in that window is the current frame.

Your goal is not to narrate the full execution. Your goal is to infer the primitive the robot is currently executing, or should execute now, from these images.

Important rules:
- Historical keyframes are always earlier than the recent visual window.
- If there is no keyframe in the recent window, keyframe_positions must be an empty list.
- keyframe_positions are 1-indexed positions within the recent 5-frame window.
- Output strict JSON only, with no extra text.
- The JSON must contain exactly two fields: current_primitive and keyframe_positions."""

SYSTEM_PROMPT_MEMORY_LONGTASK = SYSTEM_PROMPT_MEMORY_DEMO

SYSTEM_PROMPT_MEMORY = (
    SYSTEM_PROMPT_MEMORY_LONGTASK
    if os.environ.get("VLM_LONGTASK_PROMPT", "0") == "1"
    else SYSTEM_PROMPT_MEMORY_DEMO
)


@dataclass(frozen=True)
class TaskInfo:
    task_id: int
    suite: str
    task_name: str
    memory_type: str
    challenge: str
    brief_description: str
    task_block: str
    scene_description: str
    primitive_labels: list[str]
    primitive_stems: list[str]


@dataclass
class EpisodeMetrics:
    executed_steps: int = 0
    total_env_steps: int = 0
    vla_chunks: int = 0
    prompt_switches: int = 0
    final_prompt: str = ""
    final_primitive_idx: int = -1


class OfficialScoreTracker:
    """Tracks the official stage, goal, and counting-pour success signals."""

    def __init__(self, *, task_id: int, fail_on_extra_pour: bool, extra_pour_monitor_steps: int) -> None:
        self.task_id = int(task_id)
        self.counting_pour_task = stage_eval._is_counting_pour_task(self.task_id)
        self.drawer_task = stage_eval._is_drawer_task(self.task_id)
        self.fail_on_extra_pour = bool(fail_on_extra_pour)
        self.extra_pour_monitor_steps = int(extra_pour_monitor_steps)
        self.extra_pour_check = stage_eval._extra_pour_check(self.task_id)
        self.extra_monitor_start_state_idx: int | None = None
        self.extra_monitor_deadline_t: int | None = None
        self.extra_pour_detected = False
        self.pour_1_step: int | None = None
        self.pour_2_step: int | None = None

    def on_stage_done(self, *, name: str, t: int, state: dict[str, Any], logger: logging.Logger) -> None:
        if name.endswith("_Pour_One"):
            self.pour_1_step = int(t)
        elif name.endswith("_Pour_Two"):
            self.pour_2_step = int(t)
            if self.counting_pour_task and self.fail_on_extra_pour:
                self.extra_monitor_start_state_idx = int(state["step_idx"])
                self.extra_monitor_deadline_t = int(t) + self.extra_pour_monitor_steps
                logger.info("[t=%s] extra-pour monitor started; deadline=%s", t, self.extra_monitor_deadline_t)

    def observe(self, *, env: Any, state: dict[str, Any], t: int, logger: logging.Logger) -> None:
        if (
            self.counting_pour_task
            and self.fail_on_extra_pour
            and self.extra_pour_check is not None
            and self.extra_monitor_start_state_idx is not None
            and self.extra_monitor_deadline_t is not None
            and self.pour_2_step is not None
            and self.pour_2_step < t <= self.extra_monitor_deadline_t
            and self.extra_pour_check(env, state, self.extra_monitor_start_state_idx)
        ):
            self.extra_pour_detected = True
            logger.info("[t=%s] third pour detected; episode failed", t)

    def monitor_complete(self, t: int) -> bool:
        return not self.fail_on_extra_pour or (
            self.extra_monitor_deadline_t is not None and int(t) >= self.extra_monitor_deadline_t
        )

    def should_stop(self, *, t: int, stage_done: dict[str, bool]) -> bool:
        if not self.counting_pour_task:
            return False
        all_stages_complete = bool(stage_done) and all(stage_done.values())
        return self.extra_pour_detected or (all_stages_complete and self.monitor_complete(t))

    def finalize(self, *, t: int, stage_done: dict[str, bool], goal_success: bool | None) -> dict[str, Any]:
        stage_success = stage_eval._stage_success_from_stage_done(self.task_id, stage_done)
        if self.counting_pour_task:
            stage_success = stage_success and self.monitor_complete(t) and not self.extra_pour_detected
            official_success = stage_success
            strict_task_success = stage_success
            reported_goal_success: bool | None = None
        elif self.drawer_task:
            official_success = stage_success
            strict_task_success = stage_success
            reported_goal_success = stage_success
        else:
            official_success = bool(goal_success)
            strict_task_success = bool(stage_success and goal_success)
            reported_goal_success = bool(goal_success)

        if self.extra_pour_detected:
            failure_reason = "extra_pour"
        elif not stage_success:
            failure_reason = "incomplete_stage"
        elif self.counting_pour_task and not self.monitor_complete(t):
            failure_reason = "monitor_incomplete"
        elif not official_success:
            failure_reason = "goal_incomplete"
        else:
            failure_reason = None
        return {
            "stage_success": bool(stage_success),
            "goal_success": reported_goal_success,
            "official_success": bool(official_success),
            "strict_task_success": bool(strict_task_success),
            "extra_pour_detected": bool(self.extra_pour_detected),
            "pour_1_step": self.pour_1_step,
            "pour_2_step": self.pour_2_step,
            "extra_monitor_end_step": (
                self.extra_monitor_deadline_t
                if self.extra_monitor_deadline_t is not None and int(t) >= self.extra_monitor_deadline_t
                else None
            ),
            "failure_reason": failure_reason,
        }


def load_task_infos(path: Path) -> dict[int, TaskInfo]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, TaskInfo] = {}
    for task in raw["tasks"]:
        out[int(task["task_id"])] = TaskInfo(
            task_id=int(task["task_id"]),
            suite=str(task["suite"]),
            task_name=str(task["task_name"]),
            memory_type=str(task["memory_type"]),
            challenge=str(task["challenge"]),
            brief_description=str(task["brief_description"]),
            task_block=str(task["task_block"]),
            scene_description=str(task.get("scene_description", "")),
            primitive_labels=[str(p["label"]) for p in task["primitive_order"]],
            primitive_stems=[str(p.get("stem", "")) for p in task["primitive_order"]],
        )
    return out


def _camera_order_text(use_wrist_images: bool) -> str:
    if use_wrist_images:
        return (
            "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb. "
            "agentview_rgb is the external main-view camera, and eye_in_hand_rgb is the wrist/end-effector camera."
        )
    return "Camera: agentview_rgb. agentview_rgb is the external main-view camera."


def _parse_output_no_mapping(output_text: str, max_pos: int) -> tuple[str, list[int]]:
    """Parse VLM JSON output without any task-specific vocabulary mapping."""
    s = output_text.strip()
    if "</think>" in s:
        s = s[s.rfind("</think>") + len("</think>"):].strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    primitive = ""
    keyframe_positions: list[int] = []
    try:
        parsed = json.loads(s)
        primitive = str(parsed.get("current_primitive", parsed.get("current_subtask", ""))).strip()
        raw_positions = parsed.get("keyframe_positions", [])
        if isinstance(raw_positions, list):
            for p in raw_positions:
                try:
                    pi = int(p)
                except Exception:
                    continue
                if 1 <= pi <= max_pos:
                    keyframe_positions.append(pi)
    except Exception:
        # Keep raw decoded text as-is when JSON parsing fails.
        primitive = s

    return primitive, keyframe_positions


class FullVlm26MemoryPlanner(SyncLoRAPlanner):
    def __init__(self, *args: Any, task_info: TaskInfo, **kwargs: Any) -> None:
        from transformers import AutoProcessor

        processor_model_dir = kwargs.pop("processor_model_dir", None)
        if processor_model_dir is None:
            processor_model_dir = os.environ.get("VLM_PROCESSOR_DIR", kwargs.get("base_model_dir", args[0] if args else ""))
        model_dir = Path(kwargs.get("base_model_dir", args[0] if args else ""))
        processor_dir = Path(processor_model_dir)
        if model_dir.is_dir() and processor_dir.is_dir():
            for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
                src = processor_dir / name
                dst = model_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        super().__init__(*args, **kwargs)
        # Some DeepSpeed/Trainer checkpoints save model weights but not the image processor files.
        # Keep the trained weights from base_model_dir, but use the canonical Qwen3-VL processor.
        self.processor = AutoProcessor.from_pretrained(
            processor_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.set_task_info(task_info)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.task_info = task_info
        self.default_subtask_prompt = task_info.brief_description.strip()
        self._current_subtask = self.default_subtask_prompt

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None):
        super().reset_episode(instruction=instruction, run_dir=run_dir, logger=logger)
        self._current_subtask = self.default_subtask_prompt

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Image.Image | None],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
    ):
        use_wrist_images = self.use_wrist and any(
            frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
        )
        num_history_keyframes = len(memory_main_frames)
        num_history_images = num_history_keyframes * (2 if use_wrist_images else 1)
        num_context_frames = len(context_main_frames)
        num_context_images = num_context_frames * (2 if use_wrist_images else 1)

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Global objective: infer the robot's current primitive action from historical keyframes before the current step and recent visual history within the same execution.\n\n"
                    "Task objective:\n"
                    f"{self.task_info.task_block}\n\n"
                    "Scene description:\n"
                    f"{self.task_info.scene_description or self.task_info.brief_description}\n\n"
                    f"{_camera_order_text(use_wrist_images)}\n"
                    "Current observation:"
                ),
            }
        ]

        def append_timestep_images(main_frames, wrist_frames) -> None:
            for idx, main_img in enumerate(main_frames):
                user_content.append({"type": "image", "image": main_img})
                if use_wrist_images:
                    wrist_img = wrist_frames[idx] if idx < len(wrist_frames) else None
                    if wrist_img is not None:
                        user_content.append({"type": "image", "image": wrist_img})

        if memory_main_frames:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Historical keyframes from moments before the current step in the same execution "
                        f"({num_history_keyframes} timesteps, {num_history_images} images):"
                    ),
                }
            )
            append_timestep_images(memory_main_frames, memory_wrist_frames)

        user_content.append(
            {
                "type": "text",
                "text": (
                    "Recent visual context: "
                    f"{num_context_frames} consecutive frames ending at the current frame "
                    f"({num_context_images} images):"
                ),
            }
        )
        append_timestep_images(context_main_frames, context_wrist_frames)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                    "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."
                ),
            }
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def infer_sync(self, step_idx: int, context_frames_np: list[tuple[np.ndarray, np.ndarray | None]]) -> str:
        if not context_frames_np:
            return self._current_subtask

        recent_start = step_idx - len(context_frames_np) + 1
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Image.Image | None] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_np, wrist_np = frame_pack
            main_img = Image.fromarray(main_np.astype(np.uint8))
            wrist_img = Image.fromarray(wrist_np.astype(np.uint8)) if self.use_wrist and wrist_np is not None else None
            self.frame_store_main[abs_idx] = main_img
            self.frame_store_wrist[abs_idx] = wrist_img
            context_main_frames.append(main_img)
            context_wrist_frames.append(wrist_img)
        self.step = max(self.step, step_idx + 1)

        memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
        memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
        memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
        messages = self._build_messages(memory_main_frames, memory_wrist_frames, context_main_frames, context_wrist_frames)

        images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                images.extend(c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image")

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        inputs = self.processor(text=[text], images=images if images else None, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with __import__("torch").inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
        out_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        vlm_subtask, j_rel = _parse_output_no_mapping(
            out_text,
            max_pos=len(context_main_frames),
        )
        j_abs = [recent_start + (p - 1) for p in j_rel]

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
            raw_k_indices = build_visual_memory(self.J_hist, t=self.step, N=len(context_main_frames), d=self.d_merge)
            self.K_indices_abs = [idx for idx in raw_k_indices if idx < recent_start]
            self.K_main_frames = get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
            self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]
            if self.k_max > 0 and len(self.K_indices_abs) > self.k_max:
                self.K_indices_abs = self.K_indices_abs[-self.k_max:]
                self.K_main_frames = self.K_main_frames[-self.k_max:]
                self.K_wrist_frames = self.K_wrist_frames[-self.k_max:]

        self._dump_new_keyframes()
        if vlm_subtask:
            self._current_subtask = vlm_subtask
        subtask = self._current_subtask

        image_rel = None
        if self.run_dir is not None:
            image_rel = self._save_vlm_input_bundle(
                step_idx=step_idx,
                memory_main_frames=memory_main_frames,
                memory_wrist_frames=memory_wrist_frames,
                memory_indices=memory_indices,
                context_main_frames=context_main_frames,
                context_wrist_frames=context_wrist_frames,
                subtask=subtask,
            )
        self._append_trace(
            {
                "t": int(step_idx),
                "task_id": int(self.task_info.task_id),
                "subtask": subtask,
                "keyframe_positions": j_rel,
                "J_abs": j_abs,
                "K_indices_abs": list(self.K_indices_abs),
                "out_text": out_text.strip()[:600],
                "image": image_rel,
            }
        )
        if self.logger:
            self.logger.info("VLM @t=%s task=%s subtask=%s keyframes=%s", step_idx, self.task_info.task_id, subtask, j_rel)
            self.logger.info("  raw=%s", out_text.strip()[:220])
        return subtask


_OBJECT_ALIASES: tuple[tuple[str, str], ...] = (
    ("chocolate pudding", "chocolate_pudding_1"),
    ("cream cheese", "cream_cheese_1"),
    ("orange juice", "orange_juice_1"),
    ("tomato sauce", "tomato_sauce_1"),
    ("wine bottle", "wine_bottle_1"),
    ("chocolate", "chocolate_pudding_1"),
    ("pudding", "chocolate_pudding_1"),
    ("tomato", "tomato_sauce_1"),
    ("cookies", "cookies_1"),
    ("cookie", "cookies_1"),
    ("butter", "butter_1"),
    ("popcorn", "popcorn_1"),
    ("cream", "cream_cheese_1"),
    ("milk", "milk_1"),
    ("wine", "wine_bottle_1"),
)


def _raw_libero_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _has_body(env: Any, object_name: str) -> bool:
    return ec._resolve_body_id(env, object_name) is not None


def _primitive_action(label: str, stem: str) -> str:
    text = f"{label} {stem}".strip().lower().replace("_", " ")
    if text.startswith("pick "):
        return "pick"
    for action in ("place", "put", "pour", "open", "close"):
        if text.startswith(action):
            return action
    return text.split()[0] if text.split() else ""


def _primitive_stage_index(task_info: TaskInfo, primitive_idx: int) -> int | None:
    """Maps a primitive to its CSR stage index; pick primitives have no CSR stage."""
    primitive_idx = int(primitive_idx)
    if primitive_idx < 0 or primitive_idx >= len(task_info.primitive_labels):
        return None
    action = _primitive_action(task_info.primitive_labels[primitive_idx], task_info.primitive_stems[primitive_idx])
    if action == "pick":
        return None
    return sum(
        1
        for idx in range(primitive_idx)
        if _primitive_action(task_info.primitive_labels[idx], task_info.primitive_stems[idx]) != "pick"
    )


def _primitive_object_name(env: Any, label: str, stem: str) -> str | None:
    text = f"{label} {stem}".lower().replace("_", " ")
    for phrase, object_name in _OBJECT_ALIASES:
        if phrase in text and _has_body(env, object_name):
            return object_name
    return None


def _initial_body_pos_from_state(state: dict[str, Any], name: str) -> np.ndarray | None:
    initial_body_pos = state.get("initial_body_pos", {})
    variants = [name]
    if not name.endswith("_main"):
        variants.append(f"{name}_main")
    if name.endswith("_main"):
        variants.append(name[:-5])
    for cand in variants:
        if cand in initial_body_pos:
            return np.asarray(initial_body_pos[cand], dtype=np.float32)
    return None


def _is_grasped_or_lifted(env: Any, state: dict[str, Any], object_name: str) -> bool:
    raw_env = _raw_libero_env(env)
    try:
        obj = raw_env.get_object(object_name)
        robots = getattr(raw_env, "robots", None) or getattr(env, "robots", None)
        gripper = robots[0].gripper if robots else None
        if obj is not None and gripper is not None and hasattr(raw_env, "_check_grasp"):
            if bool(raw_env._check_grasp(gripper, obj)):
                return True
    except Exception:
        pass

    cur = ec._body_pos(env, object_name)
    init = _initial_body_pos_from_state(state, object_name)
    if cur is None or init is None:
        return False
    return float(cur[2] - init[2]) > 0.035


class GTSubtaskOracle:
    """Online oracle that emits short primitive-label prompts from task metadata."""

    def __init__(self, task_info: TaskInfo) -> None:
        self.set_task_info(task_info)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.task_info = task_info
        self.primitive_labels = [p.strip() for p in task_info.primitive_labels if p.strip()]
        self.primitive_stems = list(task_info.primitive_stems)
        self.default_subtask_prompt = (
            self.primitive_labels[0] if self.primitive_labels else task_info.brief_description.strip()
        )
        self.reset_episode()

    def reset_episode(self) -> None:
        self.primitive_idx = 0
        self._last_reason = "init"

    @property
    def current_primitive_idx(self) -> int:
        if not self.primitive_labels:
            return -1
        return min(self.primitive_idx, len(self.primitive_labels) - 1)

    def current_prompt(self) -> str:
        if not self.primitive_labels:
            return self.default_subtask_prompt
        return self.primitive_labels[self.current_primitive_idx]

    def _current_stem(self) -> str:
        idx = self.current_primitive_idx
        if idx < 0 or idx >= len(self.primitive_stems):
            return ""
        return self.primitive_stems[idx]

    def _advance(self, n: int, reason: str, logger: logging.Logger) -> bool:
        if not self.primitive_labels:
            return False
        old_idx = self.current_primitive_idx
        old_prompt = self.current_prompt()
        self.primitive_idx = min(len(self.primitive_labels) - 1, self.primitive_idx + max(1, n))
        new_idx = self.current_primitive_idx
        new_prompt = self.current_prompt()
        if new_idx != old_idx:
            self._last_reason = reason
            logger.info(
                "GT prompt advance [%s]: %s:%s -> %s:%s",
                reason,
                old_idx,
                old_prompt,
                new_idx,
                new_prompt,
            )
            return True
        return False

    def observe_step(
        self,
        *,
        env: Any,
        state: dict[str, Any],
        stage_completed: bool,
        logger: logging.Logger,
    ) -> bool:
        if not self.primitive_labels:
            return False
        label = self.current_prompt()
        stem = self._current_stem()
        action = _primitive_action(label, stem)

        if action == "pick":
            object_name = _primitive_object_name(env, label, stem)
            if object_name and _is_grasped_or_lifted(env, state, object_name):
                return self._advance(1, f"picked:{object_name}", logger)
            if stage_completed:
                logger.warning(
                    "GT pick prompt did not observe grasp for %s; keeping current primitive "
                    "despite CSR stage completion",
                    object_name or label,
                )
            return False

        if stage_completed:
            return self._advance(1, f"stage:{action or 'unknown'}", logger)
        return False


def _task1_stage_specs() -> list[stage_eval.StageSpec]:
    return [
        stage_eval.StageSpec(
            "01_Place_Cookies_Basket",
            stage_eval._in_container_body("cookies_1", "basket_1", 0.12, -0.05, 0.20),
        ),
        stage_eval.StageSpec(
            "02_Place_Tomato_Basket",
            stage_eval._in_container_body("tomato_sauce_1", "basket_1", 0.12, -0.05, 0.20),
        ),
    ]


def _task_specs(task_id: int) -> list[stage_eval.StageSpec]:
    if task_id == 1:
        return _task1_stage_specs()
    if task_id == 2:
        return [
            stage_eval.StageSpec(
                "01_Place_Butter_Basket",
                stage_eval._in_container_body("butter_1", "basket_1", 0.12, -0.05, 0.20),
            ),
            stage_eval.StageSpec(
                "02_Place_Popcorn_Basket",
                stage_eval._in_container_body("popcorn_1", "basket_1", 0.12, -0.05, 0.20),
            ),
        ]
    if task_id == 3:
        return [
            stage_eval.StageSpec(
                "01_Place_Cream_Basket",
                stage_eval._in_container_body("cream_cheese_1", "basket_1", 0.12, -0.05, 0.20),
            ),
            stage_eval.StageSpec(
                "02_Place_Pudding_Basket",
                stage_eval._in_container_body("chocolate_pudding_1", "basket_1", 0.12, -0.05, 0.20),
            ),
        ]
    return stage_eval._task_specs(task_id)


def _transition_task_specs(task_id: int) -> list[stage_eval.StageSpec]:
    """Primitive-aligned checks used only by the GT-subtask switching oracle.

    Official counting-pour scoring intentionally excludes setup and cleanup
    primitives. Those reduced score stages must not shift the online prompt
    boundaries, so transition checks retain the complete primitive sequence.
    """
    specs = _task_specs(task_id)
    if not stage_eval._is_counting_pour_task(task_id):
        return specs
    transition_specs: dict[int, list[stage_eval.StageSpec]] = {
        6: [
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("tomato_sauce_1", "body", "cookies_1")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("tomato_sauce_1", "body", "cookies_1")),
            stage_eval.StageSpec("Place_Bowl_Drainer", stage_eval._in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ],
        7: [
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("tomato_sauce_1", "site", "frypan_1_default_site")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("tomato_sauce_1", "site", "frypan_1_default_site")),
            stage_eval.StageSpec("Place_Bowl_Drainer", stage_eval._in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ],
        8: [
            stage_eval.StageSpec("Place_Pudding_Frypan", stage_eval._in_container_body("chocolate_pudding_1", "frypan_1", 0.10, -0.05, 0.15)),
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("tomato_sauce_1", "body", "chocolate_pudding_1")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("tomato_sauce_1", "body", "chocolate_pudding_1")),
            stage_eval.StageSpec("Place_Bowl_Drainer", stage_eval._in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ],
        9: [
            stage_eval.StageSpec("Place_Butter_Frypan", stage_eval._in_container_body("butter_1", "frypan_1", 0.10, -0.05, 0.15)),
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("tomato_sauce_1", "body", "butter_1")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("tomato_sauce_1", "body", "butter_1")),
            stage_eval.StageSpec("Place_Bowl_Drainer", stage_eval._in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ],
        10: [
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("wine_bottle_1", "site", "white_yellow_mug_1_default_site")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("wine_bottle_1", "site", "white_yellow_mug_1_default_site")),
            stage_eval.StageSpec("Place_Wine_On_Table", stage_eval._table_return("wine_bottle_1", 0.35)),
        ],
        15: [
            stage_eval.StageSpec("Place_Butter_Frypan", stage_eval._in_container_body("butter_1", "frypan_1", 0.12, -0.05, 0.15)),
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("milk_1", "body", "butter_1")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("milk_1", "body", "butter_1")),
            stage_eval.StageSpec("Place_Milk_Table", stage_eval._table_return("milk_1", 0.40)),
        ],
        16: [
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("milk_1", "site", "red_coffee_mug_1_default_site")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("milk_1", "site", "red_coffee_mug_1_default_site")),
            stage_eval.StageSpec("Place_Bowl_Drainer", stage_eval._in_container_body("milk_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ],
        22: [
            stage_eval.StageSpec("Pour_One", stage_eval._body_pour_stage("tomato_sauce_1", "body", "cookies_1")),
            stage_eval.StageSpec("Pour_Two", stage_eval._body_pour_stage("tomato_sauce_1", "body", "cookies_1")),
            stage_eval.StageSpec("Place_Tomato_Aside", stage_eval._near_fixed_position("tomato_sauce_1", np.array([0.0, -0.2, 0.50], dtype=np.float32), 0.20, 0.20)),
            stage_eval.StageSpec("Open_Microwave", stage_eval._microwave_open(0.30)),
            stage_eval.StageSpec("Place_Cookies_Microwave", stage_eval._in_microwave("cookies_1")),
            stage_eval.StageSpec("Close_Microwave", stage_eval._microwave_closed(0.05)),
        ],
    }
    return transition_specs[int(task_id)]


def _goal_override_check(task_id: int):
    if task_id == 1:
        return None
    return stage_eval._goal_override_check(task_id)


def run_episode_async_stateful(
    *,
    task_id: int,
    env: Any,
    client: Any,
    planner: FullVlm26MemoryPlanner,
    args: BaseArgs,
    stage_specs: list[stage_eval.StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
) -> tuple[float, dict[str, bool], bool, dict[str, Any], list[np.ndarray], list[np.ndarray], EpisodeMetrics]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    metrics = EpisodeMetrics(final_primitive_idx=-1)
    recent_vlm_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    worker_error: list[str] = []
    worker_stop = threading.Event()
    vlm_job_queue: queue.Queue | None = queue.Queue(maxsize=max(1, args.vlm_queue_size)) if args.async_vlm else None
    subtask_lock = threading.Lock()
    subtask_buffer = {"value": "", "step_idx": -1}
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    current_stage_start = 0
    current_subtask_prompt = ""
    last_prompt_sent: str | None = None
    goal_success = False
    ever_goal_success = False
    score_tracker = OfficialScoreTracker(
        task_id=task_id,
        fail_on_extra_pour=fail_on_extra_pour,
        extra_pour_monitor_steps=extra_pour_monitor_steps,
    )

    def raw_goal_success() -> bool:
        if score_tracker.counting_pour_task:
            return False
        if score_tracker.drawer_task:
            return stage_eval._stage_success_from_stage_done(task_id, stage_done)
        if goal_check_override is not None:
            return bool(goal_check_override(env, stage_done))
        return ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False

    def write_subtask(step_idx: int, subtask: str) -> None:
        with subtask_lock:
            subtask_buffer["value"] = subtask
            subtask_buffer["step_idx"] = step_idx

    def read_subtask() -> tuple[str, int]:
        with subtask_lock:
            return str(subtask_buffer["value"]), int(subtask_buffer["step_idx"])

    def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]

    def submit_vlm_job(step_idx: int) -> None:
        if not args.async_vlm or vlm_job_queue is None:
            return
        if step_idx < 0 or len(recent_vlm_frames) < args.n_recent:
            return
        if args.vlm_interval > 1 and step_idx % args.vlm_interval != 0:
            return
        payload = (step_idx, clone_recent_frames())
        try:
            vlm_job_queue.put_nowait(payload)
            return
        except queue.Full:
            try:
                vlm_job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                vlm_job_queue.put_nowait(payload)
            except queue.Full:
                return

    def vlm_worker() -> None:
        assert vlm_job_queue is not None
        while not worker_stop.is_set():
            try:
                payload = vlm_job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            step_idx, frames = payload
            try:
                subtask = planner.infer_sync(step_idx=step_idx, context_frames_np=frames)
                if subtask:
                    write_subtask(step_idx, subtask)
            except Exception as exc:
                worker_error.append(f"{type(exc).__name__}: {exc}")
                logger.error("VLM worker failed", exc_info=True)
                break

    vlm_thread = None
    if args.async_vlm:
        vlm_thread = threading.Thread(target=vlm_worker, name=f"vlm-task{planner.task_info.task_id}", daemon=True)
        vlm_thread.start()
        logger.info("VLM background planning enabled: single-slot subtask buffer")

    try:
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if worker_error:
                raise RuntimeError(worker_error[-1])

            if t < args.num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                metrics.total_env_steps = t
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if state is None:
                state = stage_eval._build_initial_state(env)
                current_stage_start = state["step_idx"]

            effective_t = t - args.num_steps_wait
            if len(recent_vlm_frames) < args.n_recent:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                metrics.total_env_steps = t
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if args.async_vlm:
                submit_vlm_job(effective_t)
                latest_subtask, latest_step = read_subtask()
            else:
                latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
                latest_step = effective_t

            if latest_subtask and latest_subtask != current_subtask_prompt:
                current_subtask_prompt = latest_subtask
                logger.info("[t=%s] VLM prompt update from step=%s: %s", t, latest_step, current_subtask_prompt)

            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if last_prompt_sent is not None and prompt_for_vla != last_prompt_sent:
                metrics.prompt_switches += 1
            last_prompt_sent = prompt_for_vla
            metrics.vla_chunks += 1
            metrics.final_prompt = prompt_for_vla
            logger.info("[t=%s] VLA chunk prompt=%s", t, prompt_for_vla)

            for action in actions[: args.replan_steps]:
                element_step = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                replay.append(element_step["observation/image"])
                wrist = element_step.get("observation/wrist_image")
                if wrist is not None:
                    replay_wrist.append(wrist)

                obs, _, done, _ = env.step(action.tolist())
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                if state is not None:
                    stage_eval._update_state(obs, state)
                t += 1
                metrics.executed_steps += 1
                metrics.total_env_steps = t
                submit_vlm_job(t - args.num_steps_wait)

                if state is not None and stage_idx < len(stage_specs):
                    spec = stage_specs[stage_idx]
                    if spec.check_fn(env, state, current_stage_start):
                        stage_done[spec.name] = True
                        logger.info("[t=%s] stage done: %s", t, spec.name)
                        score_tracker.on_stage_done(name=spec.name, t=t, state=state, logger=logger)
                        stage_idx += 1
                        current_stage_start = state["step_idx"]

                if stage_idx >= len(stage_specs) and not all_stages_logged:
                    logger.info("[t=%s] all stages done", t)
                    all_stages_logged = True

                score_tracker.observe(env=env, state=state, t=t, logger=logger)
                goal_success = raw_goal_success()
                if goal_success and not score_tracker.counting_pour_task:
                    if not ever_goal_success:
                        logger.info("[t=%s] goal success", t)
                    ever_goal_success = True
                if score_tracker.should_stop(t=t, stage_done=stage_done):
                    raise StopIteration
                if done or t >= args.max_steps + args.num_steps_wait:
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")
        raise
    finally:
        if args.async_vlm and vlm_job_queue is not None:
            worker_stop.set()
            try:
                vlm_job_queue.put_nowait(None)
            except queue.Full:
                pass
            if vlm_thread is not None and vlm_thread.is_alive():
                vlm_thread.join(timeout=3.0)

    stage_pct = stage_eval._stage_score_pct(task_id, stage_done)
    if not ever_goal_success and not score_tracker.counting_pour_task:
        ever_goal_success = raw_goal_success()
    diagnostics = score_tracker.finalize(t=t, stage_done=stage_done, goal_success=ever_goal_success)
    if not metrics.final_prompt:
        metrics.final_prompt = current_subtask_prompt or planner.default_subtask_prompt
    metrics.total_env_steps = t if "t" in locals() else metrics.total_env_steps
    return stage_pct, stage_done, diagnostics["strict_task_success"], diagnostics, replay, replay_wrist, metrics


def run_episode_gt_subtask(
    *,
    task_id: int,
    env: Any,
    client: Any,
    oracle: GTSubtaskOracle,
    args: BaseArgs,
    stage_specs: list[stage_eval.StageSpec],
    transition_stage_specs: list[stage_eval.StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    logger: logging.Logger,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
) -> tuple[float, dict[str, bool], bool, dict[str, Any], list[np.ndarray], list[np.ndarray], EpisodeMetrics]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    metrics = EpisodeMetrics()
    stage_done = {spec.name: False for spec in stage_specs}
    csr_stage_idx = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    csr_stage_start = 0
    primitive_stage_start = 0
    goal_success = False
    ever_goal_success = False
    last_prompt_sent: str | None = None
    score_tracker = OfficialScoreTracker(
        task_id=task_id,
        fail_on_extra_pour=fail_on_extra_pour,
        extra_pour_monitor_steps=extra_pour_monitor_steps,
    )

    def raw_goal_success() -> bool:
        if score_tracker.counting_pour_task:
            return False
        if score_tracker.drawer_task:
            return stage_eval._stage_success_from_stage_done(task_id, stage_done)
        if goal_check_override is not None:
            return bool(goal_check_override(env, stage_done))
        return ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False

    try:
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                t += 1
                metrics.total_env_steps = t
                continue

            if state is None:
                state = stage_eval._build_initial_state(env)
                csr_stage_start = state["step_idx"]
                primitive_stage_start = state["step_idx"]

            prompt_for_vla = oracle.current_prompt()
            element = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if last_prompt_sent is not None and prompt_for_vla != last_prompt_sent:
                metrics.prompt_switches += 1
            last_prompt_sent = prompt_for_vla
            metrics.vla_chunks += 1
            metrics.final_prompt = prompt_for_vla
            metrics.final_primitive_idx = oracle.current_primitive_idx
            logger.info("[t=%s] VLA chunk GT prompt=%s primitive_idx=%s", t, prompt_for_vla, oracle.current_primitive_idx)

            for action in actions[: args.replan_steps]:
                element_step = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                replay.append(element_step["observation/image"])
                wrist = element_step.get("observation/wrist_image")
                if wrist is not None:
                    replay_wrist.append(wrist)

                obs, _, done, _ = env.step(action.tolist())
                if state is not None:
                    stage_eval._update_state(obs, state)
                t += 1
                metrics.executed_steps += 1
                metrics.total_env_steps = t

                stage_completed = False
                if state is not None and csr_stage_idx < len(stage_specs):
                    spec = stage_specs[csr_stage_idx]
                    if spec.check_fn(env, state, csr_stage_start):
                        stage_done[spec.name] = True
                        stage_completed = True
                        logger.info("[t=%s] stage done: %s", t, spec.name)
                        score_tracker.on_stage_done(name=spec.name, t=t, state=state, logger=logger)
                        csr_stage_idx += 1
                        csr_stage_start = state["step_idx"]

                if csr_stage_idx >= len(stage_specs) and not all_stages_logged:
                    logger.info("[t=%s] all stages done", t)
                    all_stages_logged = True

                if state is not None:
                    primitive_stage_completed = False
                    primitive_stage_idx = _primitive_stage_index(
                        oracle.task_info,
                        oracle.current_primitive_idx,
                    )
                    if primitive_stage_idx is not None and primitive_stage_idx < len(transition_stage_specs):
                        primitive_spec = transition_stage_specs[primitive_stage_idx]
                        primitive_stage_completed = primitive_spec.check_fn(env, state, primitive_stage_start)
                    primitive_advanced = oracle.observe_step(
                        env=env,
                        state=state,
                        stage_completed=primitive_stage_completed,
                        logger=logger,
                    )
                    if primitive_advanced:
                        primitive_stage_start = state["step_idx"]
                    metrics.final_prompt = oracle.current_prompt()
                    metrics.final_primitive_idx = oracle.current_primitive_idx

                score_tracker.observe(env=env, state=state, t=t, logger=logger)
                goal_success = raw_goal_success()
                if goal_success and not score_tracker.counting_pour_task:
                    if not ever_goal_success:
                        logger.info("[t=%s] goal success", t)
                    ever_goal_success = True
                if score_tracker.should_stop(t=t, stage_done=stage_done):
                    raise StopIteration
                if done or t >= args.max_steps + args.num_steps_wait:
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")
        raise

    stage_pct = stage_eval._stage_score_pct(task_id, stage_done)
    if not ever_goal_success and not score_tracker.counting_pour_task:
        ever_goal_success = raw_goal_success()
    diagnostics = score_tracker.finalize(t=t, stage_done=stage_done, goal_success=ever_goal_success)
    if not metrics.final_prompt:
        metrics.final_prompt = oracle.current_prompt()
    metrics.final_primitive_idx = oracle.current_primitive_idx
    metrics.total_env_steps = t if "t" in locals() else metrics.total_env_steps
    return stage_pct, stage_done, diagnostics["strict_task_success"], diagnostics, replay, replay_wrist, metrics


def patch_env_resolution() -> None:
    base_env = ec._get_env_class()
    orig_init = base_env.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["camera_heights"] = 480
        kwargs["camera_widths"] = 640
        return orig_init(self, *args, **kwargs)

    base_env.__init__ = patched_init
    ec._get_env_class = lambda: base_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboMem tasks with VLM or GT subtask prompts.")
    parser.add_argument(
        "--planner-mode",
        choices=("vlm", "gt_subtask"),
        default=os.environ.get("PLANNER_MODE", "vlm"),
        help="vlm uses Qwen planning; gt_subtask skips VLM and feeds short GT primitive prompts.",
    )
    return parser.parse_args()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tsv_safe(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _run_episode_compat(runner, *, task_id: int, kwargs: dict[str, Any]):
    """Calls updated runners while keeping historical eval variants importable."""
    supported = inspect.signature(runner).parameters
    result = runner(**{key: value for key, value in kwargs.items() if key in supported})
    if len(result) == 7:
        return result
    stage_pct, stage_done, task_success, replay, replay_wrist, metrics = result
    counting_pour_task = stage_eval._is_counting_pour_task(task_id)
    stage_success = stage_eval._stage_success_from_stage_done(task_id, stage_done)
    diagnostics = {
        "stage_success": bool(stage_success),
        "goal_success": None if counting_pour_task else bool(task_success),
        "official_success": bool(stage_success if counting_pour_task else task_success),
        "strict_task_success": bool(task_success),
        "extra_pour_detected": False,
        "pour_1_step": None,
        "pour_2_step": None,
        "extra_monitor_end_step": None,
        "failure_reason": None if task_success else "legacy_runner_incomplete",
    }
    return stage_pct, stage_done, bool(task_success), diagnostics, replay, replay_wrist, metrics


def main() -> None:
    cli_args = parse_args()
    planner_mode = str(cli_args.planner_mode)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    patch_env_resolution()

    out_root = Path(os.environ["OUT_ROOT"])
    video_root = Path(os.environ["VIDEO_DIR"])
    summary_json = Path(os.environ["SUMMARY_JSON"])
    summary_tsv = Path(os.environ["SUMMARY_TSV"])
    prompt_trace_tsv = Path(os.environ.get("PROMPT_TRACE_TSV", str(out_root / "prompt_trace.tsv")))
    episode_metrics_jsonl = Path(os.environ.get("EPISODE_METRICS_JSONL", str(out_root / "episode_metrics.jsonl")))
    task_config = Path(os.environ.get("TASK_CONFIG", str(REFERENCE_DIR / "fullvlm_v2_26_memory_tasks.json")))
    task_infos = load_task_infos(task_config)
    default_tasks = list(range(1, 27)) if planner_mode == "gt_subtask" else list(range(2, 27))
    tasks = [int(x) for x in json.loads(os.environ.get("TASKS_JSON", json.dumps(default_tasks)))]
    if planner_mode == "vlm" and any(task_id == 1 for task_id in tasks):
        raise ValueError("Task 1 is intentionally excluded from this 25-task reference. Use eval_task1_nomap_reference.py for the Task 1 no-map minimal reference.")
    missing_tasks = [task_id for task_id in tasks if task_id not in task_infos]
    if missing_tasks:
        raise ValueError(f"TASKS_JSON contains task ids missing from {task_config}: {missing_tasks}")

    args = BaseArgs()
    args.host = os.environ.get("HOST", "127.0.0.1")
    args.port = int(os.environ.get("PORT", "8026"))
    args.base_model_dir = os.environ["VLM_CKPT"] if planner_mode == "vlm" else os.environ.get("VLM_CKPT", "gt_subtask")
    args.lora_path = os.environ.get("VLM_LORA_PATH", "none")
    args.vlm_device = os.environ.get("VLM_DEVICE", "cuda:1")
    args.resize_size = int(os.environ.get("RESIZE_SIZE", "256"))
    args.replan_steps = int(os.environ.get("REPLAN_STEPS", "5"))
    args.num_steps_wait = int(os.environ.get("NUM_STEPS_WAIT", "10"))
    args.max_steps = int(os.environ.get("MAX_STEPS", "2000"))
    args.seed = int(os.environ.get("SEED", "42"))
    args.num_trials_per_task = int(os.environ.get("NUM_TRIALS", "1"))
    args.vlm_input_profile = os.environ.get("VLM_INPUT_PROFILE", "fullvlm_256")
    args.vlm_match_training_jpeg_roundtrip = _env_flag("VLM_MATCH_TRAINING_JPEG_ROUNDTRIP", "0")
    args.vlm_training_jpeg_quality = int(os.environ.get("VLM_TRAINING_JPEG_QUALITY", "30"))
    args.async_vlm = _env_flag("ASYNC_VLM", "0")
    args.vlm_interval = int(os.environ.get("VLM_INTERVAL", "5"))
    args.vlm_queue_size = int(os.environ.get("VLM_QUEUE_SIZE", "1"))
    args.n_recent = int(os.environ.get("N_RECENT", "5"))
    args.k_max = int(os.environ.get("K_MAX", "0"))
    args.d_merge = int(os.environ.get("D_MERGE", "6"))
    args.vlm_use_wrist = _env_flag("VLM_USE_WRIST", "1")
    args.vlm_use_keyframe_memory = _env_flag("VLM_USE_KEYFRAME_MEMORY", "1")
    fail_on_extra_pour = _env_flag("FAIL_ON_EXTRA_POUR", "1")
    extra_pour_monitor_steps = int(os.environ.get("POST_STAGE_STEPS", os.environ.get("EXTRA_POUR_MONITOR_STEPS", "30")))
    _apply_vlm_input_profile(args)

    out_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    prompt_trace_tsv.write_text(
        "task_id\ttrial\tseed\tplanner_mode\tckpt_or_mode\tvla_prompt_last\ttsr\tcsr\t"
        "stage_success\tgoal_success\tofficial_success\textra_pour_detected\tfailure_reason\t"
        "executed_steps\ttotal_env_steps\tvla_chunks\tprompt_switches\tfinal_primitive_idx\n",
        encoding="utf-8",
    )
    summary_tsv.write_text(
        "task_id\tstatus\terror\tcsr\ttsr\tstage_success_rate\tgoal_success_rate\tofficial_success_rate\tvideo_dir\tduration_sec\t"
        "avg_executed_steps\tavg_total_env_steps\tavg_vla_chunks\tavg_prompt_switches\tavg_success_executed_steps\n",
        encoding="utf-8",
    )
    episode_metrics_jsonl.write_text("", encoding="utf-8")

    _seed_everywhere(args.seed)
    client = StableWebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None, close_timeout=30.0)
    if not tasks:
        raise ValueError("TASKS_JSON is empty; provide task ids from 1 to 26.")
    first_task = task_infos[tasks[0]]
    planner: FullVlm26MemoryPlanner | None = None
    oracle: GTSubtaskOracle | None = None
    if planner_mode == "vlm":
        planner = FullVlm26MemoryPlanner(
            base_model_dir=args.base_model_dir,
            lora_path=args.lora_path,
            instruction="",
            system_prompt=SYSTEM_PROMPT_MEMORY,
            prompt_profile="task1_kf5",
            n_recent=args.n_recent,
            d_merge=args.d_merge,
            k_max=args.k_max,
            use_keyframe_memory=args.vlm_use_keyframe_memory,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "256")),
            device=args.vlm_device,
            logger=None,
            vlm_model_type=args.vlm_model_type,
            enable_thinking=False,
            crop_right_half=False,
            use_wrist=args.vlm_use_wrist,
            task_info=first_task,
        )
    else:
        oracle = GTSubtaskOracle(first_task)
        logging.info("GT-subtask oracle mode enabled; Qwen/VLM planner is not initialized.")

    results = []
    for task_id in tasks:
        task_info = task_infos[task_id]
        if planner is not None:
            planner.set_task_info(task_info)
        if oracle is not None:
            oracle.set_task_info(task_info)
        bddl_path = ec._resolve_bddl_path(task_id)
        stage_specs = _task_specs(task_id)
        transition_stage_specs = _transition_task_specs(task_id)
        counting_pour_task = stage_eval._is_counting_pour_task(task_id)
        goal_monitor_dict = {} if counting_pour_task else ec._build_goal_monitor_dict(bddl_path)
        goal_check_override = _goal_override_check(task_id)
        task_video = video_root / f"task{task_id}"
        task_video.mkdir(parents=True, exist_ok=True)
        task_root = out_root / f"task{task_id}"
        task_root.mkdir(parents=True, exist_ok=True)
        status = "completed"
        err = ""
        st = time.time()
        stage_sum = 0.0
        task_success_cnt = 0
        goal_cnt = 0
        goal_scored_episodes = 0
        stage_success_cnt = 0
        official_success_cnt = 0
        executed_steps_values: list[int] = []
        total_env_steps_values: list[int] = []
        vla_chunks_values: list[int] = []
        prompt_switch_values: list[int] = []
        success_executed_steps_values: list[int] = []

        try:
            env_cls = ec._get_env_class()
            env = None
            last_env_init_exc: Exception | None = None
            for attempt in range(1, ENV_INIT_RETRIES + 1):
                try:
                    env = env_cls(
                        bddl_file_name=str(bddl_path),
                        camera_heights=480,
                        camera_widths=640,
                        ignore_done=True,
                        reward_shaping=True,
                        control_freq=20,
                        initialization_noise=None,
                    )
                    if attempt > 1:
                        logging.info(
                            "Env init recovered at attempt %s/%s for task%s",
                            attempt,
                            ENV_INIT_RETRIES,
                            task_id,
                        )
                    break
                except Exception as exc:
                    if not stage_eval._is_randomization_error(exc):  # noqa: SLF001
                        raise
                    last_env_init_exc = exc
                    logging.warning(
                        "Env init randomization failed (%s/%s) for task%s: %s",
                        attempt,
                        ENV_INIT_RETRIES,
                        task_id,
                        exc,
                    )
                    if attempt < ENV_INIT_RETRIES:
                        time.sleep(ENV_INIT_RETRY_SLEEP)
            if env is None:
                raise RuntimeError(
                    f"Env init failed after {ENV_INIT_RETRIES} retries for task{task_id} "
                    f"(last error: {last_env_init_exc})"
                )
            for ep in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task{task_id}"):
                seed = args.seed + ep
                _seed_everywhere(seed)
                try:
                    env.seed(seed)
                except AttributeError:
                    pass
                run_dir = task_root / f"ep{ep}"
                ep_logger = make_episode_logger(run_dir)
                ep_logger.info("task_id=%s bddl=%s planner_mode=%s ckpt_or_mode=%s", task_id, bddl_path, planner_mode, args.base_model_dir)
                if planner_mode == "vlm":
                    assert planner is not None
                    planner.reset_episode(instruction="", run_dir=run_dir, logger=ep_logger)
                    stage_pct, stage_done, task_success, diagnostics, replay, replay_wrist, metrics = _run_episode_compat(
                        run_episode_async_stateful,
                        task_id=task_id,
                        kwargs={
                            "task_id": task_id,
                            "env": env,
                            "client": client,
                            "planner": planner,
                            "args": args,
                            "stage_specs": stage_specs,
                            "goal_monitor_dict": goal_monitor_dict,
                            "goal_check_override": goal_check_override,
                            "vlm_camera_pose": None,
                            "logger": ep_logger,
                            "fail_on_extra_pour": fail_on_extra_pour,
                            "extra_pour_monitor_steps": extra_pour_monitor_steps,
                        },
                    )
                else:
                    assert oracle is not None
                    oracle.reset_episode()
                    stage_pct, stage_done, task_success, diagnostics, replay, replay_wrist, metrics = _run_episode_compat(
                        run_episode_gt_subtask,
                        task_id=task_id,
                        kwargs={
                            "task_id": task_id,
                            "env": env,
                            "client": client,
                            "oracle": oracle,
                            "args": args,
                            "stage_specs": stage_specs,
                            "transition_stage_specs": transition_stage_specs,
                            "goal_monitor_dict": goal_monitor_dict,
                            "goal_check_override": goal_check_override,
                            "logger": ep_logger,
                            "fail_on_extra_pour": fail_on_extra_pour,
                            "extra_pour_monitor_steps": extra_pour_monitor_steps,
                        },
                    )
                stage_sum += stage_pct
                task_success_cnt += int(task_success)
                stage_success_cnt += int(diagnostics["stage_success"])
                official_success_cnt += int(diagnostics["official_success"])
                if diagnostics["goal_success"] is not None:
                    goal_cnt += int(diagnostics["goal_success"])
                    goal_scored_episodes += 1
                executed_steps_values.append(metrics.executed_steps)
                total_env_steps_values.append(metrics.total_env_steps)
                vla_chunks_values.append(metrics.vla_chunks)
                prompt_switch_values.append(metrics.prompt_switches)
                if task_success:
                    success_executed_steps_values.append(metrics.executed_steps)
                base_name = ec.get_video_basename(task_id, ep, seed, diagnostics["official_success"])
                stages_str = " | ".join(f"{k}={'Y' if v else 'N'}" for k, v in stage_done.items())
                ep_logger.info(
                    "Episode %s seed=%s CSR=%.1f TSR=%s steps=%s chunks=%s switches=%s prompt=%s | %s",
                    ep,
                    seed,
                    stage_pct,
                    int(task_success),
                    metrics.executed_steps,
                    metrics.vla_chunks,
                    metrics.prompt_switches,
                    metrics.final_prompt,
                    stages_str,
                )
                with prompt_trace_tsv.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{task_id}\t{ep}\t{seed}\t{planner_mode}\t{_tsv_safe(args.base_model_dir)}\t"
                        f"{_tsv_safe(metrics.final_prompt)}\t{int(task_success)}\t{stage_pct:.1f}\t"
                        f"{int(diagnostics['stage_success'])}\t"
                        f"{'' if diagnostics['goal_success'] is None else int(diagnostics['goal_success'])}\t"
                        f"{int(diagnostics['official_success'])}\t{int(diagnostics['extra_pour_detected'])}\t"
                        f"{_tsv_safe(diagnostics['failure_reason'])}\t"
                        f"{metrics.executed_steps}\t{metrics.total_env_steps}\t{metrics.vla_chunks}\t"
                        f"{metrics.prompt_switches}\t{metrics.final_primitive_idx}\n"
                    )
                with episode_metrics_jsonl.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "trial": ep,
                                "seed": seed,
                                "planner_mode": planner_mode,
                                "csr": stage_pct,
                                "tsr": bool(task_success),
                                "stage_score_pct": stage_pct,
                                "stage_success": bool(diagnostics["stage_success"]),
                                "goal_success": diagnostics["goal_success"],
                                "official_success": bool(diagnostics["official_success"]),
                                "extra_pour_detected": bool(diagnostics["extra_pour_detected"]),
                                "pour_1_step": diagnostics["pour_1_step"],
                                "pour_2_step": diagnostics["pour_2_step"],
                                "extra_monitor_end_step": diagnostics["extra_monitor_end_step"],
                                "failure_reason": diagnostics["failure_reason"],
                                "stage_done": stage_done,
                                "executed_steps": metrics.executed_steps,
                                "total_env_steps": metrics.total_env_steps,
                                "vla_chunks": metrics.vla_chunks,
                                "prompt_switches": metrics.prompt_switches,
                                "final_prompt": metrics.final_prompt,
                                "final_primitive_idx": metrics.final_primitive_idx,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if replay:
                    try:
                        _write_video(task_video / f"{base_name}.mp4", replay, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write main video for %s", base_name)
                if replay_wrist:
                    try:
                        _write_video(task_video / f"{base_name}_wrist.mp4", replay_wrist, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write wrist video for %s", base_name)
            env.close()
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        n = max(1, args.num_trials_per_task)
        csr = stage_sum / n
        tsr = task_success_cnt / n
        stage_success_rate = stage_success_cnt / n
        goal_success_rate = None if goal_scored_episodes == 0 else goal_cnt / goal_scored_episodes
        official_success_rate = official_success_cnt / n
        dur = round(time.time() - st, 2)
        avg_executed_steps = _avg(executed_steps_values)
        avg_total_env_steps = _avg(total_env_steps_values)
        avg_vla_chunks = _avg(vla_chunks_values)
        avg_prompt_switches = _avg(prompt_switch_values)
        avg_success_executed_steps = _avg(success_executed_steps_values)
        row = {
            "task_id": task_id,
            "status": status,
            "error": err,
            "csr": csr,
            "tsr": tsr,
            "stage_score_pct": csr,
            "stage_success_rate": stage_success_rate,
            "goal_success_rate": goal_success_rate,
            "official_success_rate": official_success_rate,
            "video_dir": str(task_video),
            "duration_sec": dur,
            "avg_executed_steps": avg_executed_steps,
            "avg_total_env_steps": avg_total_env_steps,
            "avg_vla_chunks": avg_vla_chunks,
            "avg_prompt_switches": avg_prompt_switches,
            "avg_success_executed_steps": avg_success_executed_steps,
        }
        results.append(row)
        summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        with summary_tsv.open("a", encoding="utf-8") as f:
            goal_rate_text = "N/A" if goal_success_rate is None else f"{goal_success_rate:.4f}"
            f.write(
                f"{task_id}\t{status}\t{_tsv_safe(err)}\t{csr:.1f}\t{tsr:.4f}\t"
                f"{stage_success_rate:.4f}\t{goal_rate_text}\t{official_success_rate:.4f}\t{task_video}\t{dur}\t"
                f"{'' if avg_executed_steps is None else f'{avg_executed_steps:.2f}'}\t"
                f"{'' if avg_total_env_steps is None else f'{avg_total_env_steps:.2f}'}\t"
                f"{'' if avg_vla_chunks is None else f'{avg_vla_chunks:.2f}'}\t"
                f"{'' if avg_prompt_switches is None else f'{avg_prompt_switches:.2f}'}\t"
                f"{'' if avg_success_executed_steps is None else f'{avg_success_executed_steps:.2f}'}\n"
            )
        logging.info(
            "task=%s status=%s CSR=%.1f TSR=%.3f stage_success=%.3f goal=%s official_success=%.3f avg_steps=%s",
            task_id,
            status,
            csr,
            tsr,
            stage_success_rate,
            goal_rate_text,
            official_success_rate,
            avg_executed_steps,
        )

    if planner is not None:
        planner.close()
    completed = [r for r in results if r["status"] == "completed"]
    goal_scored = [r for r in completed if r["goal_success_rate"] is not None]
    total_csr = sum(r["csr"] for r in completed) / max(1, len(completed))
    total_tsr = sum(r["tsr"] for r in completed) / max(1, len(completed))
    aggregate = {
        "macro_csr": total_csr,
        "macro_tsr": total_tsr,
        "macro_stage_score_pct": total_csr,
        "macro_stage_success_rate": sum(r["stage_success_rate"] for r in completed) / max(1, len(completed)),
        "macro_goal_success_rate": sum(r["goal_success_rate"] for r in goal_scored) / max(1, len(goal_scored)),
        "macro_official_success_rate": sum(r["official_success_rate"] for r in completed) / max(1, len(completed)),
        "num_tasks": len(results),
        "num_goal_scored_tasks": len(goal_scored),
        "planner_mode": planner_mode,
        "avg_executed_steps": _avg([r["avg_executed_steps"] for r in completed if r["avg_executed_steps"] is not None]),
        "avg_total_env_steps": _avg([r["avg_total_env_steps"] for r in completed if r["avg_total_env_steps"] is not None]),
        "avg_vla_chunks": _avg([r["avg_vla_chunks"] for r in completed if r["avg_vla_chunks"] is not None]),
        "avg_prompt_switches": _avg([r["avg_prompt_switches"] for r in completed if r["avg_prompt_switches"] is not None]),
        "avg_success_executed_steps": _avg([r["avg_success_executed_steps"] for r in completed if r["avg_success_executed_steps"] is not None]),
    }
    (out_root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("done macro CSR=%.1f TSR=%.3f summary=%s", total_csr, total_tsr, summary_tsv)
    failed = [r for r in results if r["status"] != "completed"]
    if failed:
        raise SystemExit(f"Evaluation failed for {len(failed)} task(s); see {summary_tsv}")


if __name__ == "__main__":
    main()
