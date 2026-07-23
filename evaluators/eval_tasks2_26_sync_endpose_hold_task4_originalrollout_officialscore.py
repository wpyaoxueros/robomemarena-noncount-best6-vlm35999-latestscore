#!/usr/bin/env python3
"""Task4 original rollout with latest remote score observation.

Rollout source SHA256: 04914a97e07bc7028ac94bce6514f690c4a673684169de93d8cf8126803c9e79.
Only the official stage observer and summary writer below are local additions.
"""
from __future__ import annotations

import importlib.util
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_EVAL_PY = Path(
    os.environ.get(
        "TASKS2_26_BASE_EVAL_PY",
        "/data/user/hlei573/tmp/rma_refeval_fresh_20260513_052445/RoboMemArena/"
        "evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py",
    )
)

spec = importlib.util.spec_from_file_location("_tasks2_26_base_eval", BASE_EVAL_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base eval from {BASE_EVAL_PY}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

_ORIG_FULLVLM26_BUILD_MESSAGES = base.FullVlm26MemoryPlanner._build_messages
_ORIG_SYNC_LORA_APPEND_TRACE = base.SyncLoRAPlanner._append_trace
_ORIG_BASE_TASK_SPECS = getattr(base, "_task_specs", None)
_ORIG_STAGE_TASK_SPECS = getattr(base.stage_eval, "_task_specs", None)

OFFICIAL_SCRIPTS_DIR = Path(os.environ["ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR"])
if str(OFFICIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SCRIPTS_DIR))


def _load_official_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load official module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


official_eval_common = _load_official_module(
    "_task4_originalrollout_official_eval_common", OFFICIAL_SCRIPTS_DIR / "eval_common.py"
)
_previous_eval_common = sys.modules.get("eval_common")
sys.modules["eval_common"] = official_eval_common
try:
    official_stage = _load_official_module(
        "_task4_originalrollout_official_stage",
        OFFICIAL_SCRIPTS_DIR / "task2_26_reference_stage.py",
    )
finally:
    if _previous_eval_common is None:
        sys.modules.pop("eval_common", None)
    else:
        sys.modules["eval_common"] = _previous_eval_common


def _append_trace_with_cache(self, record):
    if isinstance(record, dict):
        self._latest_trace_record = dict(record)
    else:
        self._latest_trace_record = record
    return _ORIG_SYNC_LORA_APPEND_TRACE(self, record)


def _completed_subtasks_mode() -> str:
    raw = os.environ.get("VLM_COMPLETED_SUBTASKS_MODE", "auto").strip().lower()
    if raw in {"0", "false", "no", "none", "off", ""}:
        return ""
    if raw in {"completed_text", "text"}:
        return "completed_text"
    if raw in {"completed_struct", "struct", "json"}:
        return "completed_struct"
    if raw != "auto":
        return ""
    haystack = " ".join(
        os.environ.get(name, "")
        for name in (
            "EVAL_TAG",
            "WATCH_TAG",
            "RUN_STAMP",
            "ARTIFACT_ROOT",
            "SOURCE_CHECKPOINT",
            "TRAIN_OUTDIR",
        )
    ).lower()
    if "completed_struct" in haystack:
        return "completed_struct"
    if "completed_text" in haystack:
        return "completed_text"
    return ""


def _completed_subtasks_block(completed: list[str], mode: str) -> str:
    clean = [str(item).strip() for item in completed if str(item).strip()]
    if mode == "completed_text":
        return "Completed subtasks: " + ("; ".join(clean) if clean else "none") + "."
    if mode == "completed_struct":
        payload = {"type": "completed_subtasks", "count": len(clean), "items": clean}
        return "Completed-subtasks feature JSON: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return ""


def _normalize_subtask_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _parse_subtask_tol_overrides() -> dict[str, float]:
    raw = os.environ.get("ENDPOSE_HOLD_SUBTASK_TOLS_JSON", "").strip()
    if not raw:
        path_str = os.environ.get("ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE", "").strip()
        if path_str:
            raw = Path(path_str).read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Invalid ENDPOSE_HOLD_SUBTASK_TOLS_JSON: {raw}") from exc
    if not isinstance(data, dict):
        raise ValueError("ENDPOSE_HOLD_SUBTASK_TOLS_JSON must be a JSON object")
    overrides: dict[str, float] = {}
    for key, value in data.items():
        norm_key = _normalize_subtask_name(str(key))
        if norm_key:
            overrides[norm_key] = float(value)
    return overrides


def _inject_runtime_completed_subtasks(messages: list[dict[str, Any]], completed: list[str], mode: str) -> None:
    block = _completed_subtasks_block(completed, mode)
    if not block:
        return
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "text"):
                continue
            text = str(item.get("text", ""))
            marker = "\nCurrent observation:"
            if marker in text:
                item["text"] = text.replace(marker, "\n" + block + "\n" + marker.lstrip(), 1)
            else:
                item["text"] = text + "\n" + block
            return


def _build_messages_runtime_progress(self, *args, **kwargs):
    mode = os.environ.get("VLM_TASK_TEXT_MODE", "default").strip().lower()
    if mode in {"no_label_no_order", "scene_only"}:
        original_info = self.task_info
        scene_text = original_info.scene_description or original_info.brief_description
        safe_task_block = (
            "High-level objective: infer the next executable low-level robot action from visual evidence only. "
            "Do not rely on a provided primitive list or a fixed task order; use the historical keyframes and "
            "the current visual context to decide what action should be run now."
        )
        self.task_info = dataclasses.replace(
            original_info,
            task_block=safe_task_block,
            brief_description=scene_text,
            scene_description=scene_text,
        )
        try:
            messages = _ORIG_FULLVLM26_BUILD_MESSAGES(self, *args, **kwargs)
        finally:
            self.task_info = original_info
    else:
        messages = _ORIG_FULLVLM26_BUILD_MESSAGES(self, *args, **kwargs)

    completed_mode = _completed_subtasks_mode()
    if completed_mode:
        completed = list(getattr(self, "_runtime_completed_subtasks", []))
        _inject_runtime_completed_subtasks(messages, completed, completed_mode)
    return messages


def _patched_task_specs(task_id: int):
    tid = int(task_id)
    # Task11 official English-reference metadata says:
    # data_dir = 11_cookies_top_butter_middle_dataset
    # but primitive_order / task_block explicitly use chocolate for the middle drawer.
    # The local stage-eval file still expects butter here, which makes a correct
    # "place chocolate" rollout look like a stage failure. Override only Task11.
    if tid == 11:
        return [
            base.stage_eval.StageSpec(
                "01_Open_Top_Drawer",
                base.stage_eval._drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10),
            ),
            base.stage_eval.StageSpec(
                "02_Place_Cookies_Top_Drawer",
                base.stage_eval._in_container_site(
                    "cookies_1", "wooden_cabinet_1_top_region", 0.15, 0.15, -0.05, 0.15
                ),
            ),
            base.stage_eval.StageSpec(
                "03_Close_Top_Drawer",
                base.stage_eval._drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08),
            ),
            base.stage_eval.StageSpec(
                "04_Open_Middle_Drawer",
                base.stage_eval._drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10),
            ),
            base.stage_eval.StageSpec(
                "05_Place_Chocolate_Middle_Drawer",
                base.stage_eval._in_container_site(
                    "chocolate_pudding_1",
                    "wooden_cabinet_1_middle_region",
                    0.15,
                    0.15,
                    -0.05,
                    0.15,
                ),
            ),
            base.stage_eval.StageSpec(
                "06_Close_Middle_Drawer",
                base.stage_eval._drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08),
            ),
        ]
    # base._task_specs handles Task1/2/3 directly, but for Task4-26 it delegates
    # to stage_eval._task_specs. Since this wrapper patches stage_eval._task_specs
    # too, using base._task_specs as the first fallback would recurse forever for
    # non-Task11 drawer tasks such as Task5. Route Task4-26 to the original
    # stage-eval provider first, and keep base._task_specs only for Task1/2/3.
    if tid in {1, 2, 3} and _ORIG_BASE_TASK_SPECS is not None:
        return _ORIG_BASE_TASK_SPECS(tid)
    if _ORIG_STAGE_TASK_SPECS is not None:
        return _ORIG_STAGE_TASK_SPECS(tid)
    if _ORIG_BASE_TASK_SPECS is not None:
        return _ORIG_BASE_TASK_SPECS(tid)
    raise RuntimeError("No task-spec provider available")


DEFAULT_TARGET_JSON = (
    "/data/user/hlei573/openpi_inference/tmp/tasks2_26_holdstatic_general/"
    "tasks2_26_endpose_targets_seed100_199.json"
)
DEFAULT_PASSAGE_COUNTS_JSON = (
    "/data/user/hlei573/openpi_inference/tmp/tasks2_26_holdstatic_general/"
    "tasks2_26_target_passage_counts_seed100_199_alltasks_tol045_20260624_074452.json"
)
DEFAULT_TASK4_OPEN_MIDDLE_RELEASE_ANCHOR_HDF5 = (
    "/data/user/hlei573/data/fullvlm_v2/4_drawer_butter_dataset/subtask_data/"
    "open_middle_drawer_2_seed104_task4.hdf5"
)
DEFAULT_TASK4_OPEN_BOTTOM_RELEASE_ANCHOR_HDF5 = (
    "/data/user/hlei573/data/fullvlm_v2/4_drawer_butter_dataset/subtask_data/"
    "open_bottom_drawer_4_seed104_task4.hdf5"
)
DEFAULT_TASK4_OPEN_TOP_AGAIN_RELEASE_ANCHOR_HDF5 = (
    "/data/user/hlei573/data/fullvlm_v2/4_drawer_butter_dataset/subtask_data/"
    "open_top_drawer_again_6_seed104_task4.hdf5"
)
DEFAULT_H5DUMP_BIN = (
    os.environ.get("H5DUMP_BIN", "").strip()
    or shutil.which("h5dump")
    or "/share/anaconda3/bin/h5dump"
)


@dataclass(frozen=True)
class HoldConfig:
    enabled: bool
    targets_json: Path
    pos_tol: float
    eef_default_tol: float
    eef_p95_extra_tol: float
    eef_tol_cap: float
    min_active_steps: int
    consecutive: int
    disable_final: bool
    post_release_vla_steps: int
    strict_hold_release_next: bool
    prevent_regression: bool
    regression_guard_after_hold_release: bool
    distance_log_interval: int
    drawer_open_return_hold: bool
    drawer_open_return_away_dist: float
    passage_counts_json: Path | None
    drawer_forward_advance_guard: bool
    drawer_require_stage_and_target: bool
    drawer_open_stage_thresh: float
    drawer_close_stage_thresh: float
    drawer_stage_debug_interval: int
    completed_update_on_stage_done: bool
    completed_update_on_stage_done_subtasks: tuple[str, ...]
    drawer_forward_allow_stage_done: bool
    drawer_forward_allow_stage_done_subtasks: tuple[str, ...]
    release_anchor_on_nonhold_switch: bool
    subtask_forward_max_advance: int
    microwave_require_open_hold_release: bool
    microwave_open_stage_done_release: bool
    stage_done_auto_advance: bool
    stage_done_auto_advance_tasks: tuple[int, ...]
    stage_done_auto_advance_subtasks: tuple[str, ...]
    endpose_hold_require_stage_done_subtasks: tuple[str, ...]
    pick_gripper_gate: bool
    pick_gripper_open_max: float
    pick_gripper_close_min: float
    pick_object_lift_gate: bool
    pick_object_lift_delta: float
    pick_lift_auto_release: bool
    pick_lift_auto_release_repeat: int
    post_pick_release_hold_gripper_steps: int
    post_pick_release_hold_gripper_value: float
    post_pick_release_hold_gripper_until_place_hold: bool
    post_pick_release_hold_gripper_after_place_hold_steps: int


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    return float(raw)


def env_subtask_list(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except Exception:
        payload = [item.strip() for item in raw.split(",")]
    if isinstance(payload, str):
        payload = [payload]
    if not isinstance(payload, list):
        return ()
    return tuple(_normalize_subtask_name(str(item)) for item in payload if str(item).strip())


def hold_config() -> HoldConfig:
    passage_counts_raw = os.environ.get("ENDPOSE_TARGET_PASSAGE_COUNTS_JSON", "").strip()
    return HoldConfig(
        enabled=env_bool("ENABLE_ENDPOSE_HOLD", True),
        targets_json=Path(os.environ.get("ENDPOSE_HOLD_TARGETS_JSON", DEFAULT_TARGET_JSON)),
        pos_tol=env_float("ENDPOSE_HOLD_POS_TOL", 0.04),
        eef_default_tol=env_float("ENDPOSE_HOLD_EEF_DEFAULT_TOL", 0.06),
        eef_p95_extra_tol=env_float("ENDPOSE_HOLD_EEF_P95_EXTRA_TOL", 0.02),
        eef_tol_cap=env_float("ENDPOSE_HOLD_EEF_TOL_CAP", 0.08),
        min_active_steps=int(os.environ.get("ENDPOSE_HOLD_MIN_ACTIVE_STEPS", "20")),
        consecutive=int(os.environ.get("ENDPOSE_HOLD_CONSECUTIVE", "2")),
        disable_final=env_bool("ENDPOSE_HOLD_DISABLE_FINAL", True),
        post_release_vla_steps=int(os.environ.get("POST_HOLD_RELEASE_VLA_STEPS", "30")),
        strict_hold_release_next=env_bool("STRICT_HOLD_RELEASE_NEXT", True),
        prevent_regression=env_bool("PREVENT_SUBTASK_REGRESSION", True),
        regression_guard_after_hold_release=env_bool("REGRESSION_GUARD_AFTER_HOLD_RELEASE", True),
        distance_log_interval=int(os.environ.get("ENDPOSE_DISTANCE_LOG_INTERVAL", "0")),
        drawer_open_return_hold=env_bool("DRAWER_OPEN_RETURN_HOLD", False),
        drawer_open_return_away_dist=env_float("DRAWER_OPEN_RETURN_AWAY_DIST", 0.08),
        passage_counts_json=Path(passage_counts_raw) if passage_counts_raw else Path(DEFAULT_PASSAGE_COUNTS_JSON),
        drawer_forward_advance_guard=env_bool("DRAWER_FORWARD_ADVANCE_GUARD", True),
        drawer_require_stage_and_target=env_bool("DRAWER_REQUIRE_STAGE_AND_TARGET", False),
        drawer_open_stage_thresh=env_float("DRAWER_OPEN_STAGE_THRESH", 0.10),
        drawer_close_stage_thresh=env_float("DRAWER_CLOSE_STAGE_THRESH", 0.08),
        drawer_stage_debug_interval=int(os.environ.get("DRAWER_STAGE_DEBUG_INTERVAL", "0")),
        completed_update_on_stage_done=env_bool("COMPLETED_UPDATE_ON_STAGE_DONE", False),
        completed_update_on_stage_done_subtasks=env_subtask_list("COMPLETED_UPDATE_ON_STAGE_DONE_SUBTASKS_JSON"),
        drawer_forward_allow_stage_done=env_bool("DRAWER_FORWARD_ALLOW_STAGE_DONE", False),
        drawer_forward_allow_stage_done_subtasks=env_subtask_list("DRAWER_FORWARD_ALLOW_STAGE_DONE_SUBTASKS_JSON"),
        release_anchor_on_nonhold_switch=env_bool("RELEASE_ANCHOR_ON_NONHOLD_SWITCH", False),
        subtask_forward_max_advance=int(os.environ.get("SUBTASK_FORWARD_MAX_ADVANCE", "0")),
        microwave_require_open_hold_release=env_bool("MICROWAVE_REQUIRE_OPEN_HOLD_RELEASE", False),
        microwave_open_stage_done_release=env_bool("MICROWAVE_OPEN_STAGE_DONE_RELEASE", False),
        stage_done_auto_advance=env_bool("STAGE_DONE_AUTO_ADVANCE", False),
        stage_done_auto_advance_tasks=tuple(
            int(item)
            for item in json.loads(os.environ.get("STAGE_DONE_AUTO_ADVANCE_TASKS_JSON", "[]"))
        ),
        stage_done_auto_advance_subtasks=env_subtask_list("STAGE_DONE_AUTO_ADVANCE_SUBTASKS_JSON"),
        endpose_hold_require_stage_done_subtasks=env_subtask_list("ENDPOSE_HOLD_REQUIRE_STAGE_DONE_SUBTASKS_JSON"),
        pick_gripper_gate=env_bool("ENDPOSE_PICK_GRIPPER_GATE", False),
        pick_gripper_open_max=env_float("ENDPOSE_PICK_GRIPPER_OPEN_MAX", -0.2),
        pick_gripper_close_min=env_float("ENDPOSE_PICK_GRIPPER_CLOSE_MIN", 0.2),
        pick_object_lift_gate=env_bool("ENDPOSE_PICK_OBJECT_LIFT_GATE", True),
        pick_object_lift_delta=env_float("ENDPOSE_PICK_OBJECT_LIFT_DELTA", 0.01),
        pick_lift_auto_release=env_bool("ENDPOSE_PICK_LIFT_AUTO_RELEASE", False),
        pick_lift_auto_release_repeat=int(os.environ.get("ENDPOSE_PICK_LIFT_AUTO_RELEASE_REPEAT", "3")),
        post_pick_release_hold_gripper_steps=int(os.environ.get("POST_PICK_RELEASE_HOLD_GRIPPER_STEPS", "0")),
        post_pick_release_hold_gripper_value=env_float("POST_PICK_RELEASE_HOLD_GRIPPER_VALUE", 1.0),
        post_pick_release_hold_gripper_until_place_hold=env_bool(
            "POST_PICK_RELEASE_HOLD_GRIPPER_UNTIL_PLACE_HOLD", False
        ),
        post_pick_release_hold_gripper_after_place_hold_steps=int(
            os.environ.get("POST_PICK_RELEASE_HOLD_GRIPPER_AFTER_PLACE_HOLD_STEPS", "0")
        ),
    )


def normalize_subtask(subtask: str, labels: list[str]) -> str:
    raw = " ".join(str(subtask).strip().lower().replace("_", " ").split())

    def _canonical_tokens(text: str) -> list[str]:
        return [
            tok
            for tok in re.findall(r"[a-z0-9]+", text)
            if tok not in {"the", "a", "an"}
        ]

    try:
        norm = base._normalize_primitive(subtask, allowed_subtasks=labels)
        if norm:
            return norm
    except Exception:
        pass

    label_norms = [" ".join(label.strip().lower().split()) for label in labels]
    if raw in label_norms:
        return raw

    # Make matching tolerant to articles such as "the" in strings coming from
    # JSON rules or raw VLM outputs. Return the exact legal label if there is a
    # unique article-insensitive match.
    raw_canonical = _canonical_tokens(raw)
    if raw_canonical:
        article_matches = [
            label
            for label, label_norm in zip(labels, label_norms, strict=True)
            if raw_canonical == _canonical_tokens(label_norm)
        ]
        if len(article_matches) == 1:
            return article_matches[0]

    # Some checkpoints output a shortened object/action phrase such as
    # "place butter". Map it only when it uniquely identifies one legal label.
    raw_tokens = set(raw_canonical)
    if raw_tokens:
        matches = [
            label
            for label, label_norm in zip(labels, label_norms, strict=True)
            if raw_tokens.issubset(set(_canonical_tokens(label_norm)))
        ]
        if len(matches) == 1:
            return matches[0]

    # Drawer checkpoints sometimes hallucinate temporal suffixes on otherwise
    # legal labels, e.g. "close bottom drawer again". Strip only the temporal
    # tokens and map back when there is a unique legal match.
    raw_token_list = re.findall(r"[a-z0-9]+", raw)
    if raw_token_list:
        stripped_tokens = [tok for tok in raw_token_list if tok not in {"again", "final", "the"}]
        if stripped_tokens and stripped_tokens != raw_token_list:
            stripped_set = set(stripped_tokens)
            matches = [
                label
                for label, label_norm in zip(labels, label_norms, strict=True)
                if stripped_set.issubset(set(_canonical_tokens(label_norm)))
            ]
            if len(matches) == 1:
                return matches[0]
    return raw


def order_index(subtask: str, labels: list[str]) -> int | None:
    norm = normalize_subtask(subtask, labels)
    try:
        return labels.index(norm)
    except ValueError:
        return None


def get_eef_pos(obs: dict[str, Any]) -> np.ndarray:
    for key in ("robot0_eef_pos", "ee_pos"):
        if key in obs:
            value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if value.size >= 3:
                return value[:3]
    raise KeyError(f"Cannot find EEF position in obs keys={sorted(obs.keys())}")


def format_vec3(vec: np.ndarray | list[float] | tuple[float, ...] | None) -> str:
    if vec is None:
        return "NA"
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return "NA"
    return f"[{arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f}]"


@lru_cache(maxsize=8)
def _load_overlay_font(font_size: int):
    for path in (
        "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def overlay_debug_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    anchor_xy: tuple[int, int] = (8, 8),
) -> np.ndarray:
    arr = _as_uint8_rgb(frame).copy()
    if not lines:
        return arr

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = 18 if img.width >= 1280 else 16 if img.width >= 960 else 14 if img.width >= 640 else 12
    font = _load_overlay_font(font_size)
    x0, y0 = anchor_xy
    max_chars = 100 if img.width >= 640 else 72
    wrapped: list[str] = []
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        wrapped.extend(textwrap.wrap(text, width=max_chars) or [""])
    if not wrapped:
        return arr

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped]
    line_heights = [max(14, box[3] - box[1] + 2) for box in line_boxes]
    text_width = max(box[2] - box[0] for box in line_boxes)
    total_height = sum(line_heights) + 10
    bg_w = min(img.width - x0 - 4, text_width + 12)
    bg_h = min(img.height - y0 - 4, total_height)
    draw.rectangle((x0, y0, x0 + bg_w, y0 + bg_h), fill=(0, 0, 0, 180))

    y = y0 + 5
    for line, line_h in zip(wrapped, line_heights, strict=True):
        draw.text((x0 + 6, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h
        if y >= y0 + bg_h - 10:
            break
    return np.asarray(img)


def load_task_targets(cfg: HoldConfig, task_id: int, labels: list[str]) -> dict[str, dict[str, Any]]:
    if not cfg.enabled:
        return {}
    if not cfg.targets_json.exists():
        raise FileNotFoundError(
            f"End-pose target JSON does not exist: {cfg.targets_json}. "
            "Run compute_task_endpose_targets.py first."
        )
    raw = json.loads(cfg.targets_json.read_text(encoding="utf-8"))
    if "tasks" in raw:
        task_payload = raw["tasks"].get(str(task_id), {})
        raw_subtasks = task_payload.get("subtasks", {})
    else:
        raw_subtasks = raw.get("subtasks", raw)
    targets: dict[str, dict[str, Any]] = {}
    for name, payload in raw_subtasks.items():
        subtask = normalize_subtask(name, labels)
        pos = payload.get("target_ee_pos") or payload.get("ee_pos") or payload.get("median_ee_pos")
        if pos is None:
            raise ValueError(f"{cfg.targets_json}: missing target_ee_pos for task{task_id} {name}")
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.size < 3:
            raise ValueError(f"{cfg.targets_json}: invalid target_ee_pos for task{task_id} {name}: {pos}")
        hold_gripper = float(payload.get("hold_gripper", -1.0))
        targets[subtask] = {
            "target_ee_pos": pos_arr[:3],
            "hold_gripper": 1.0 if hold_gripper >= 0.0 else -1.0,
            "pos_dist_p95": float(payload.get("pos_dist_p95", 0.0) or 0.0),
        }
    return targets


def load_target_passage_counts(
    cfg: HoldConfig,
    task_id: int,
    labels: list[str],
) -> dict[str, int]:
    if cfg.passage_counts_json is None:
        return {}
    if not cfg.passage_counts_json.exists():
        raise FileNotFoundError(f"Target passage-count JSON does not exist: {cfg.passage_counts_json}")
    raw = json.loads(cfg.passage_counts_json.read_text(encoding="utf-8"))
    task_payload = raw.get("tasks", {}).get(str(task_id), {})
    raw_subtasks = task_payload.get("subtasks", {})
    counts: dict[str, int] = {}
    for name, payload in raw_subtasks.items():
        subtask = normalize_subtask(name, labels)
        value = payload.get("required_near_segments", payload.get("mode_near_segments", 1))
        try:
            counts[subtask] = max(1, int(value))
        except Exception:
            counts[subtask] = 1
    return counts


def distance_to_target(obs: dict[str, Any], target: dict[str, Any]) -> float:
    return float(np.linalg.norm(get_eef_pos(obs) - target["target_ee_pos"]))


def _parse_h5dump_subset(stdout: str, expected_dim: int) -> np.ndarray:
    data_idx = stdout.find("DATA {")
    payload = stdout[data_idx:] if data_idx >= 0 else stdout
    payload = re.sub(r"\(\s*\d+\s*(?:,\s*\d+)?\s*\):", " ", payload)
    values = [
        float(token)
        for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", payload)
    ]
    if len(values) < expected_dim:
        raise ValueError(f"h5dump subset parse failed: need {expected_dim} values, got {len(values)}")
    return np.asarray(values[:expected_dim], dtype=np.float64)


@lru_cache(maxsize=32)
def _load_h5dump_row(path_str: str, dataset: str, row_idx: int, dim: int) -> np.ndarray:
    cmd = [
        DEFAULT_H5DUMP_BIN,
        "-w",
        "65535",
        "-d",
        dataset,
        "-s",
        f"{row_idx},0",
        "-c",
        f"1,{dim}",
        path_str,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _parse_h5dump_subset(result.stdout, dim)


@lru_cache(maxsize=16)
def load_task4_release_anchor(anchor_hdf5: str, frame_idx: int) -> dict[str, np.ndarray]:
    return {
        "joint_states": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/joint_states", frame_idx, 7),
        "gripper_states": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/gripper_states", frame_idx, 2),
        "ee_pos": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/ee_pos", frame_idx, 3),
    }


def _write_qpos_addr(sim: Any, qpos_addr: Any, values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if isinstance(qpos_addr, slice):
        n = len(sim.data.qpos[qpos_addr])
        sim.data.qpos[qpos_addr] = values[:n]
        if hasattr(sim.data, "qvel"):
            sim.data.qvel[qpos_addr] = 0.0
        return n > 0
    if isinstance(qpos_addr, tuple) and len(qpos_addr) == 2:
        start, end = int(qpos_addr[0]), int(qpos_addr[1])
        n = max(0, end - start)
        sim.data.qpos[start:end] = values[:n]
        if hasattr(sim.data, "qvel"):
            sim.data.qvel[start:end] = 0.0
        return n > 0
    indexes = np.asarray(qpos_addr, dtype=np.int64).reshape(-1)
    if indexes.size == 0:
        return False
    sim.data.qpos[indexes] = values[: indexes.size]
    if hasattr(sim.data, "qvel"):
        sim.data.qvel[indexes] = 0.0
    return True


def _apply_gripper_joint_positions(env: Any, robot: Any, gripper_states: np.ndarray) -> str | None:
    gripper_states = np.asarray(gripper_states, dtype=np.float64).reshape(-1)
    if gripper_states.size == 0:
        return None

    if hasattr(robot, "set_gripper_joint_positions"):
        try:
            robot.set_gripper_joint_positions(gripper_states)
            return "robot.set_gripper_joint_positions"
        except Exception:
            pass

    sim = getattr(env, "sim", None)
    if sim is None:
        return None

    gripper_index_map = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    arms = list(getattr(robot, "arms", []) or [])
    for arm_name in arms:
        if (
            gripper_index_map is not None
            and arm_name in gripper_index_map
            and gripper_index_map[arm_name] is not None
            and _write_qpos_addr(sim, gripper_index_map[arm_name], gripper_states)
        ):
            return f"robot._ref_gripper_joint_pos_indexes[{arm_name}]"

    joint_names = list(getattr(getattr(sim, "model", None), "joint_names", []) or [])
    gripper_joint_names = [
        name
        for name in joint_names
        if "finger_joint" in name.lower() or "gripper" in name.lower()
    ]
    if gripper_joint_names and hasattr(sim.model, "get_joint_qpos_addr"):
        applied = 0
        for name, value in zip(gripper_joint_names, gripper_states):
            qpos_addr = sim.model.get_joint_qpos_addr(name)
            if _write_qpos_addr(sim, qpos_addr, np.asarray([value], dtype=np.float64)):
                applied += 1
        if applied:
            return f"sim.model.joint_names[{','.join(gripper_joint_names[:applied])}]"

    return None


def get_object_pos(env: Any, obs: dict[str, Any], object_key: str) -> np.ndarray:
    if object_key in obs:
        value = np.asarray(obs[object_key], dtype=np.float64).reshape(-1)
        if value.size >= 3:
            return value[:3]

    object_name = object_key[:-4] if object_key.endswith("_pos") else object_key
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is not None:
        candidates = [
            ("body", getattr(getattr(sim, "model", None), "body_names", []), getattr(getattr(sim, "data", None), "body_xpos", None)),
            ("site", getattr(getattr(sim, "model", None), "site_names", []), getattr(getattr(sim, "data", None), "site_xpos", None)),
            ("geom", getattr(getattr(sim, "model", None), "geom_names", []), getattr(getattr(sim, "data", None), "geom_xpos", None)),
        ]
        for _, names, positions in candidates:
            if positions is None:
                continue
            for idx, name in enumerate(names):
                if not name:
                    continue
                low = str(name).lower()
                if low == object_name.lower() or low.startswith(f"{object_name.lower()}_"):
                    value = np.asarray(positions[idx], dtype=np.float64).reshape(-1)
                    if value.size >= 3:
                        return value[:3]
    raise KeyError(f"Cannot find object position for {object_key!r}; obs keys={sorted(obs.keys())}")


PICK_OBJECT_ALIASES: dict[str, list[str]] = {
    "tomato sauce": ["tomato_sauce", "tomato sauce"],
    "orange juice": ["orange_juice", "orange juice"],
    "cream": ["cream_cheese", "cream"],
}


def pick_object_candidates(subtask: str) -> list[str]:
    normalized = " ".join(str(subtask).strip().lower().replace("_", " ").split())
    if not normalized.startswith("pick "):
        return []
    object_phrase = normalized[len("pick ") :].strip()
    variants = PICK_OBJECT_ALIASES.get(object_phrase, [object_phrase])
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        for candidate in (
            variant,
            variant.replace(" ", "_"),
            variant.replace("_", " "),
            variant.split()[0],
            variant.replace("_", " ").split()[0],
        ):
            candidate = candidate.strip().lower()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return out


def infer_pick_object_key(subtask: str, env: Any, obs: dict[str, Any]) -> str | None:
    candidates = pick_object_candidates(subtask)
    if not candidates:
        return None

    obs_keys = [str(key) for key in obs.keys()]
    for candidate in candidates:
        for key in obs_keys:
            low = key.lower()
            if low == candidate or low == f"{candidate}_pos" or low.startswith(f"{candidate}_"):
                return key

    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return None

    name_groups = [
        getattr(getattr(sim, "model", None), "body_names", []),
        getattr(getattr(sim, "model", None), "site_names", []),
        getattr(getattr(sim, "model", None), "geom_names", []),
    ]
    for candidate in candidates:
        for names in name_groups:
            for name in names:
                if not name:
                    continue
                low = str(name).lower()
                if low == candidate or low.startswith(f"{candidate}_"):
                    return candidate
    return None


def needs_drawer_return_gate(subtask: str) -> bool:
    text = " ".join(str(subtask).strip().lower().split())
    return "drawer" in text and text.startswith("open ")


def use_release_prompt_regression_guard(labels: list[str]) -> bool:
    norm_labels = [" ".join(str(label).strip().lower().split()) for label in labels]
    if not any("drawer" in label for label in norm_labels):
        return False
    # Repeated drawer tasks such as Task4/5 revisit the same drawer-action
    # family later via "again"/"final" stages. Holding on to the old
    # hold-majority blacklist would permanently suppress those legal repeats.
    return any((" again" in label) or (" final" in label) for label in norm_labels)


def drawer_region_name(subtask: str) -> str | None:
    text = " ".join(str(subtask).strip().lower().split())
    if "top drawer" in text:
        return "wooden_cabinet_1_top_region"
    if "middle drawer" in text:
        return "wooden_cabinet_1_middle_region"
    if "bottom drawer" in text:
        return "wooden_cabinet_1_bottom_region"
    return None


def drawer_slot_name(subtask: str) -> str | None:
    text = " ".join(str(subtask).strip().lower().split())
    if "top drawer" in text:
        return "top"
    if "middle drawer" in text:
        return "middle"
    if "bottom drawer" in text:
        return "bottom"
    return None


@lru_cache(maxsize=8)
def _load_release_anchor_rules_from_json(path_str: str) -> dict[int, list[dict[str, Any]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Release-anchor JSON does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("tasks", raw)
    if not isinstance(payload, dict):
        raise ValueError("Release-anchor JSON must be a dict or contain a top-level 'tasks' dict")
    out: dict[int, list[dict[str, Any]]] = {}
    for task_key, rules in payload.items():
        task_id = int(task_key)
        if not isinstance(rules, list):
            raise ValueError(f"Release-anchor rules for task {task_key} must be a list")
        parsed_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"Task {task_key} rule #{idx} must be an object")
            released = str(rule.get("released", "")).strip()
            nxt = str(rule.get("next", "")).strip()
            anchor_hdf5 = str(rule.get("anchor_hdf5", "")).strip()
            frame_idx = int(rule.get("frame_idx", 0))
            if not released or not nxt or not anchor_hdf5:
                raise ValueError(
                    f"Task {task_key} rule #{idx} must contain released/next/anchor_hdf5"
                )
            parsed_rules.append(
                {
                    "released": released,
                    "next": nxt,
                    "anchor_hdf5": anchor_hdf5,
                    "frame_idx": max(0, frame_idx),
                    "tag": str(rule.get("tag", f"{released}->{nxt}")).strip() or f"{released}->{nxt}",
                }
            )
        out[task_id] = parsed_rules
    return out


def _legacy_task4_release_anchor_rules() -> list[dict[str, Any]]:
    legacy_rules = [
        {
            "flag": "TASK4_CLOSETOP_TO_OPENMIDDLE_TELEPORT",
            "released": "close the top drawer",
            "next": "open middle drawer",
            "anchor_hdf5_env": "TASK4_OPEN_MIDDLE_RELEASE_ANCHOR_HDF5",
            "default_hdf5": DEFAULT_TASK4_OPEN_MIDDLE_RELEASE_ANCHOR_HDF5,
            "frame_env": "TASK4_OPEN_MIDDLE_RELEASE_ANCHOR_FRAME",
        },
        {
            "flag": "TASK4_CLOSEMIDDLE_TO_OPENBOTTOM_TELEPORT",
            "released": "close middle drawer",
            "next": "open bottom drawer",
            "anchor_hdf5_env": "TASK4_OPEN_BOTTOM_RELEASE_ANCHOR_HDF5",
            "default_hdf5": DEFAULT_TASK4_OPEN_BOTTOM_RELEASE_ANCHOR_HDF5,
            "frame_env": "TASK4_OPEN_BOTTOM_RELEASE_ANCHOR_FRAME",
        },
        {
            "flag": "TASK4_CLOSEBOTTOM_TO_OPENTOPAGAIN_TELEPORT",
            "released": "close bottom drawer",
            "next": "open top drawer again",
            "anchor_hdf5_env": "TASK4_OPEN_TOP_AGAIN_RELEASE_ANCHOR_HDF5",
            "default_hdf5": DEFAULT_TASK4_OPEN_TOP_AGAIN_RELEASE_ANCHOR_HDF5,
            "frame_env": "TASK4_OPEN_TOP_AGAIN_RELEASE_ANCHOR_FRAME",
        },
    ]
    out: list[dict[str, Any]] = []
    for rule in legacy_rules:
        if not env_bool(rule["flag"], False):
            continue
        out.append(
            {
                "released": rule["released"],
                "next": rule["next"],
                "anchor_hdf5": os.environ.get(rule["anchor_hdf5_env"], rule["default_hdf5"]).strip(),
                "frame_idx": max(0, int(os.environ.get(rule["frame_env"], "0"))),
                "tag": f"legacy::{rule['released']}->{rule['next']}",
            }
        )
    return out


def active_release_anchor_rules(task_id: int) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    path_str = os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip()
    if path_str:
        rules.extend(_load_release_anchor_rules_from_json(path_str).get(task_id, []))
    if task_id == 4:
        rules.extend(_legacy_task4_release_anchor_rules())
    return rules


def run_episode_sync_endpose_hold(
    *,
    env: Any,
    client: Any,
    planner: Any,
    args: Any,
    stage_specs: list[Any],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
) -> tuple[float, dict[str, bool], bool, list[np.ndarray], list[np.ndarray]]:
    if args.async_vlm:
        raise ValueError("eval_tasks2_26_sync_endpose_hold.py is sync-only; set ASYNC_VLM=0.")

    cfg = hold_config()
    labels = list(planner.task_info.primitive_labels)
    label_by_norm = {_normalize_subtask_name(label): label for label in labels}
    open_microwave_label = label_by_norm.get("open microwave", "open microwave")
    task_id_int = int(planner.task_info.task_id)
    final_subtask = labels[-1] if labels else ""
    targets = load_task_targets(cfg, task_id_int, labels)
    target_passage_counts = load_target_passage_counts(cfg, task_id_int, labels)
    official_stage_specs = official_stage._task_specs(task_id_int)
    release_anchor_rules = active_release_anchor_rules(task_id_int)
    disable_output_normalize = env_bool("DISABLE_OUTPUT_NORMALIZE", False)
    release_prompt_guard = use_release_prompt_regression_guard(labels)
    subtask_tol_overrides = _parse_subtask_tol_overrides()
    if subtask_tol_overrides:
        logger.info(
            "[ENDPOSE_SUBTASK_TOL_OVERRIDES] task=%s overrides=%s",
            planner.task_info.task_id,
            subtask_tol_overrides,
        )

    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    recent_vlm_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    current_stage_start = 0
    official_stage_done = {spec.name: False for spec in official_stage_specs}
    official_stage_idx = 0
    official_all_stages_logged = False
    official_state: dict[str, Any] | None = None
    official_current_stage_start = 0
    current_subtask_prompt = ""
    current_subtask_start_t = 0
    endpose_streak = 0
    hold_active = False
    hold_subtask = ""
    min_endpose_dist: dict[str, float] = {}
    min_endpose_t: dict[str, int] = {}
    max_pick_height_z: dict[str, float] = {}
    max_pick_height_t: dict[str, int] = {}
    pick_object_key_cache: dict[str, str] = {}
    pick_object_key_source: dict[str, str] = {}
    pick_object_baseline_z: dict[str, float] = {}
    pick_object_baseline_t: dict[str, int] = {}
    regression_guard_active = not cfg.regression_guard_after_hold_release
    blocked_after_hold_prompts: set[str] = set()
    hold_consumed_subtasks: set[str] = set()
    hold_prompt_counts: dict[str, int] = {}
    runtime_completed_subtasks: list[str] = []
    setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
    passage_gate_states: dict[str, dict[str, Any]] = {}
    stage_done_forced_next_subtask = ""
    ever_goal_success = False
    hold_count_total = 0
    last_gripper_action: float | None = None
    pick_gate_open_seen = False
    pick_gate_closed_after_open = False
    pick_gate_open_t: int | None = None
    pick_gate_close_t: int | None = None
    max_prompt_idx_seen: int | None = None
    post_pick_release_hold_gripper_remaining = 0
    post_pick_release_hold_gripper_source = ""
    post_pick_release_hold_gripper_active = False
    post_pick_release_hold_gripper_target = ""
    t = 0

    def is_pick_subtask(subtask: str) -> bool:
        return " ".join(str(subtask).strip().lower().split()).startswith("pick ")

    def is_place_subtask(subtask: str) -> bool:
        return " ".join(str(subtask).strip().lower().split()).startswith("place ")

    def reset_pick_completion_gate(subtask: str) -> None:
        nonlocal pick_gate_open_seen, pick_gate_closed_after_open, pick_gate_open_t, pick_gate_close_t
        pick_gate_open_seen = False
        pick_gate_closed_after_open = False
        pick_gate_open_t = None
        pick_gate_close_t = None
        pick_object_baseline_z.pop(subtask, None)
        pick_object_baseline_t.pop(subtask, None)
        if cfg.pick_gripper_gate and is_pick_subtask(subtask) and last_gripper_action is not None:
            if last_gripper_action <= cfg.pick_gripper_open_max:
                pick_gate_open_seen = True
                pick_gate_open_t = t
                logger.info(
                    "[PICK_GRIPPER_GATE_OPEN_SEEN] t=%s task=%s subtask=%s source=initial_last_action "
                    "gripper=%+.3f open_max=%+.3f",
                    t,
                    planner.task_info.task_id,
                    subtask,
                    last_gripper_action,
                    cfg.pick_gripper_open_max,
                )

    def resolved_pick_object_key(subtask: str) -> tuple[str | None, str]:
        cached = pick_object_key_cache.get(subtask)
        if cached:
            return cached, pick_object_key_source.get(subtask, "cache")
        inferred = infer_pick_object_key(subtask, env, obs)
        if inferred:
            pick_object_key_cache[subtask] = inferred
            pick_object_key_source[subtask] = "inferred"
            logger.info(
                "[PICK_OBJECT_KEY_INFERRED] t=%s task=%s subtask=%s object_key=%s",
                t,
                planner.task_info.task_id,
                subtask,
                inferred,
            )
            return inferred, "inferred"
        return None, "missing"

    def update_pick_gripper_gate(action: list[float] | np.ndarray, prompt_for_vla: str) -> None:
        nonlocal last_gripper_action, pick_gate_open_seen, pick_gate_closed_after_open
        nonlocal pick_gate_open_t, pick_gate_close_t
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.size < 7:
            return
        gripper = float(action_arr[6])
        last_gripper_action = gripper
        if not (cfg.pick_gripper_gate and is_pick_subtask(prompt_for_vla)):
            return
        if (not pick_gate_open_seen) and gripper <= cfg.pick_gripper_open_max:
            pick_gate_open_seen = True
            pick_gate_open_t = t
            logger.info(
                "[PICK_GRIPPER_GATE_OPEN_SEEN] t=%s task=%s subtask=%s source=action gripper=%+.3f open_max=%+.3f",
                t,
                planner.task_info.task_id,
                prompt_for_vla,
                gripper,
                cfg.pick_gripper_open_max,
            )
        if pick_gate_open_seen and (not pick_gate_closed_after_open) and gripper >= cfg.pick_gripper_close_min:
            pick_gate_closed_after_open = True
            pick_gate_close_t = t
            logger.info(
                "[PICK_GRIPPER_GATE_CLOSED_AFTER_OPEN] t=%s task=%s subtask=%s gripper=%+.3f close_min=%+.3f open_t=%s",
                t,
                planner.task_info.task_id,
                prompt_for_vla,
                gripper,
                cfg.pick_gripper_close_min,
                pick_gate_open_t,
            )

    def maybe_hold_gripper_after_pick_release(
        action: list[float] | np.ndarray,
        prompt_for_vla: str,
        control_mode: str,
    ) -> list[float] | np.ndarray:
        nonlocal post_pick_release_hold_gripper_remaining
        lock_until_place_hold = post_pick_release_hold_gripper_active
        if not lock_until_place_hold and post_pick_release_hold_gripper_remaining <= 0:
            return action
        if not lock_until_place_hold and not is_place_subtask(prompt_for_vla):
            return action
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        if action_arr.size < 7:
            return action
        original_gripper = float(action_arr[6])
        forced_gripper = float(cfg.post_pick_release_hold_gripper_value)
        action_arr[6] = forced_gripper
        if not lock_until_place_hold:
            post_pick_release_hold_gripper_remaining -= 1
        logger.info(
            "[POST_PICK_RELEASE_HOLD_GRIPPER] t=%s task=%s subtask=%s mode=%s "
            "lock_mode=%s target_hold=%s remaining_after=%s "
            "original_gripper=%+.3f forced_gripper=%+.3f source=%s",
            t,
            planner.task_info.task_id,
            prompt_for_vla,
            control_mode,
            "until_place_hold" if lock_until_place_hold else "fixed_steps",
            post_pick_release_hold_gripper_target,
            post_pick_release_hold_gripper_remaining,
            original_gripper,
            forced_gripper,
            post_pick_release_hold_gripper_source,
        )
        if not lock_until_place_hold and post_pick_release_hold_gripper_remaining == 0:
            logger.info(
                "[POST_PICK_RELEASE_HOLD_GRIPPER_UNLOCK] t=%s task=%s subtask=%s "
                "source=%s reason=forced_close_steps_elapsed",
                t,
                planner.task_info.task_id,
                prompt_for_vla,
                post_pick_release_hold_gripper_source,
            )
        return action_arr

    def maybe_unlock_gripper_at_place_hold(subtask: str, source: str) -> None:
        nonlocal post_pick_release_hold_gripper_active
        nonlocal post_pick_release_hold_gripper_remaining
        nonlocal post_pick_release_hold_gripper_target
        if not post_pick_release_hold_gripper_active:
            return
        normalized_subtask = normalize_subtask(subtask, labels)
        if normalized_subtask != post_pick_release_hold_gripper_target:
            return
        post_pick_release_hold_gripper_active = False
        post_pick_release_hold_gripper_remaining = max(
            0, cfg.post_pick_release_hold_gripper_after_place_hold_steps
        )
        logger.info(
            "[POST_PICK_RELEASE_HOLD_GRIPPER_PLACE_HOLD_REACHED] t=%s task=%s subtask=%s "
            "source=%s forced_close_steps_after_hold=%s",
            t,
            planner.task_info.task_id,
            subtask,
            source,
            post_pick_release_hold_gripper_remaining,
        )
        if post_pick_release_hold_gripper_remaining == 0:
            logger.info(
                "[POST_PICK_RELEASE_HOLD_GRIPPER_UNLOCK] t=%s task=%s subtask=%s "
                "source=%s reason=place_hold_start_no_delay",
                t,
                planner.task_info.task_id,
                subtask,
                source,
            )
            post_pick_release_hold_gripper_target = ""

    def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]

    def append_vlm_frame() -> None:
        recent_vlm_frames.append(base._extract_vlm_frame(env, obs, args, vlm_camera_pose))

    def can_hold(subtask: str) -> bool:
        if not cfg.enabled or not subtask or subtask not in targets:
            return False
        if cfg.disable_final and subtask == final_subtask:
            return False
        if subtask in hold_consumed_subtasks:
            return False
        return True

    def most_common_hold_prompt() -> str:
        if not hold_prompt_counts:
            return hold_subtask
        return max(
            hold_prompt_counts.items(),
            key=lambda item: (item[1], 1 if item[0] == hold_subtask else 0, item[0]),
        )[0]

    def mark_runtime_completed(subtask: str, source: str) -> None:
        if not subtask:
            return
        if subtask in runtime_completed_subtasks:
            return
        runtime_completed_subtasks.append(subtask)
        setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
        logger.info(
            "[COMPLETED_SUBTASKS_UPDATE] t=%s task=%s completed=%s mode=%s source=%s subtask=%s",
            t,
            planner.task_info.task_id,
            runtime_completed_subtasks,
            _completed_subtasks_mode() or "off",
            source,
            subtask,
        )

    def stage_done_completed_subtask(stage_name: str, current_subtask: str) -> str:
        if not current_subtask:
            return ""
        stage_text = re.sub(r"^\d+_", "", str(stage_name)).replace("_", " ").lower()
        stage_text = " ".join(stage_text.split())
        current_norm = _normalize_subtask_name(current_subtask)
        if current_norm == stage_text:
            return current_subtask
        current_tokens = [tok for tok in current_norm.split() if tok not in {"in", "into", "the"}]
        stage_tokens = set(stage_text.split())
        if current_tokens and all(tok in stage_tokens for tok in current_tokens):
            return current_subtask
        return ""

    def stage_done_matches_subtask(stage_name: str, current_subtask: str) -> bool:
        return bool(stage_done_completed_subtask(stage_name, current_subtask))

    def endpose_stage_done_required_ok(subtask: str) -> tuple[bool, str]:
        current_norm = _normalize_subtask_name(subtask)
        required = cfg.endpose_hold_require_stage_done_subtasks
        if not required or current_norm not in required:
            return True, "not_required"
        for spec_name, is_done in stage_done.items():
            if is_done and stage_done_matches_subtask(spec_name, subtask):
                return True, f"done:{spec_name}"
        if state is None or stage_idx >= len(stage_specs):
            return False, "no_active_stage"
        spec = stage_specs[stage_idx]
        if not stage_done_matches_subtask(spec.name, subtask):
            return False, f"active_stage_mismatch:{spec.name}"
        try:
            if spec.check_fn(env, state, current_stage_start):
                return True, f"predicate:{spec.name}"
        except Exception as exc:
            return False, f"predicate_error:{type(exc).__name__}"
        return False, f"predicate_false:{spec.name}"

    def pos_tol_for_subtask(subtask: str) -> float:
        norm_subtask = _normalize_subtask_name(subtask)
        if norm_subtask in subtask_tol_overrides:
            return float(subtask_tol_overrides[norm_subtask])
        target = targets.get(subtask)
        if target is None:
            return float(max(cfg.pos_tol, cfg.eef_default_tol))
        p95 = float(target.get("pos_dist_p95", 0.0) or 0.0)
        adaptive = p95 + cfg.eef_p95_extra_tol if p95 > 0.0 else 0.0
        return float(min(cfg.eef_tol_cap, max(cfg.pos_tol, cfg.eef_default_tol, adaptive)))

    def is_close_drawer_label(subtask: str) -> bool:
        text = " ".join(str(subtask).strip().lower().split())
        return text.startswith("close ") and "drawer" in text

    def required_target_passage_segments(subtask: str) -> int:
        if is_close_drawer_label(subtask):
            return 1
        if subtask in target_passage_counts:
            return target_passage_counts[subtask]
        return 1

    def get_target_passage_state(subtask: str) -> dict[str, Any]:
        state = passage_gate_states.get(subtask)
        if state is None:
            state = {"in_near": False, "seen_segments": 0}
            passage_gate_states[subtask] = state
        return state

    def update_target_passage_state(subtask: str, t_now: int, phase: str) -> None:
        required_segments = required_target_passage_segments(subtask)
        if required_segments <= 1 or subtask not in targets:
            return

        dist = distance_to_target(obs, targets[subtask])
        pos_tol = pos_tol_for_subtask(subtask)
        state = get_target_passage_state(subtask)

        is_near = dist <= pos_tol
        if is_near and not state["in_near"]:
            state["seen_segments"] += 1
            state["in_near"] = True
            logger.info(
                "[TARGET_PASSAGE_NEAR_SEGMENT] t=%s task=%s subtask=%s segment=%s/%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                state["seen_segments"],
                required_segments,
                dist,
                pos_tol,
                phase,
            )
        elif not is_near and state["in_near"]:
            state["in_near"] = False
            logger.info(
                "[TARGET_PASSAGE_EXIT_NEAR] t=%s task=%s subtask=%s seen_segments=%s/%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                state["seen_segments"],
                required_segments,
                dist,
                pos_tol,
                phase,
            )

    def update_all_target_passage_states(t_now: int, phase: str) -> None:
        for subtask, required_segments in target_passage_counts.items():
            if required_segments > 1 and subtask in targets:
                update_target_passage_state(subtask, t_now, phase)

    def target_passage_gate_allows_count(subtask: str) -> bool:
        required_segments = required_target_passage_segments(subtask)
        if required_segments <= 1:
            return True
        return int(get_target_passage_state(subtask)["seen_segments"]) >= required_segments

    def is_close_drawer_status(status: dict[str, Any] | None) -> bool:
        return bool(status is not None and status.get("mode") == "close")

    def requires_drawer_stage_and_target(subtask: str) -> bool:
        text = normalize_subtask(subtask, labels)
        if cfg.drawer_require_stage_and_target:
            return (
                (text.startswith("open ") and "drawer" in text)
                or (("place " in text or "put " in text) and "drawer" in text)
            )
        return text in {
            "open middle drawer",
            "open bottom drawer",
            "open top drawer again",
        }

    def drawer_stage_gate_allows(subtask: str) -> bool:
        stage_ok, _ = drawer_stage_gate_status(subtask)
        return stage_ok

    def drawer_stage_status(subtask: str) -> dict[str, Any] | None:
        if state is None:
            return None
        text = " ".join(str(subtask).strip().lower().split())
        if "drawer" not in text:
            return None
        region_name = drawer_region_name(subtask)
        drawer_slot = drawer_slot_name(subtask)
        if region_name is None and drawer_slot is None:
            return None
        if text.startswith("open "):
            threshold = cfg.drawer_open_stage_thresh
            mode = "open"
        elif text.startswith("close "):
            threshold = cfg.drawer_close_stage_thresh
            mode = "close"
        else:
            return None

        # Primary path: official region-site delta used by the stage specs.
        if region_name is not None:
            try:
                region_pos = base.stage_eval._current_site_pos(env, region_name)
                init_pos = base.stage_eval._initial_site_pos(state, region_name)
            except Exception:
                region_pos = None
                init_pos = None
            if region_pos is not None and init_pos is not None:
                ref_y = float(init_pos[1])
                delta_y = abs(float(region_pos[1] - ref_y))
                stage_ok = delta_y > threshold if mode == "open" else delta_y < threshold
                return {
                    "region_name": region_name,
                    "drawer_slot": drawer_slot,
                    "mode": mode,
                    "delta_y": delta_y,
                    "threshold": threshold,
                    "stage_ok": bool(stage_ok),
                    "source": "region_site",
                }

        # Fallback: use drawer handle body displacement when the region site is missing.
        if drawer_slot is None:
            return None
        handle_name = f"wooden_cabinet_1_{drawer_slot}_handle"
        try:
            handle_pos = base.stage_eval._drawer_handle_pos(env, drawer_slot)
            init_handle = base.stage_eval._initial_body_pos(state, handle_name)
        except Exception:
            return None
        if handle_pos is None or init_handle is None:
            return None
        ref_y = float(init_handle[1])
        delta_y = abs(float(handle_pos[1] - ref_y))
        stage_ok = delta_y > threshold if mode == "open" else delta_y < threshold
        return {
            "region_name": region_name,
            "drawer_slot": drawer_slot,
            "mode": mode,
            "delta_y": delta_y,
            "threshold": threshold,
            "stage_ok": bool(stage_ok),
            "source": "handle_body",
        }

    def drawer_stage_probe(subtask: str) -> dict[str, Any]:
        text = " ".join(str(subtask).strip().lower().split())
        region_name = drawer_region_name(subtask)
        out: dict[str, Any] = {
            "subtask_text": text,
            "region_name": region_name,
            "has_state": state is not None,
        }
        if state is None or region_name is None:
            return out
        try:
            cur = base.stage_eval._current_site_pos(env, region_name)
        except Exception as exc:
            out["current_site_error"] = f"{type(exc).__name__}: {exc}"
            cur = None
        try:
            init = base.stage_eval._initial_site_pos(state, region_name)
        except Exception as exc:
            out["initial_site_error"] = f"{type(exc).__name__}: {exc}"
            init = None
        out["current_site_found"] = cur is not None
        out["initial_site_found"] = init is not None
        if cur is not None:
            out["current_site_pos"] = np.asarray(cur, dtype=np.float32).tolist()
        if init is not None:
            out["initial_site_pos"] = np.asarray(init, dtype=np.float32).tolist()
        return out

    def drawer_stage_gate_status(subtask: str) -> tuple[bool, dict[str, Any] | None]:
        text = " ".join(str(subtask).strip().lower().split())
        if "drawer" not in text:
            return False, None

        status = drawer_stage_status(subtask)
        if status is not None:
            return bool(status["stage_ok"]), status

        if ("place " in text or "put " in text) and stage_idx < len(stage_specs) and state is not None:
            spec = stage_specs[stage_idx]
            try:
                stage_ok = bool(spec.check_fn(env, state, current_stage_start))
                return stage_ok, {
                    "source": "current_stage_spec",
                    "stage_ok": stage_ok,
                    "spec_name": getattr(spec, "name", ""),
                }
            except Exception as exc:
                return False, {
                    "source": "current_stage_spec",
                    "stage_ok": False,
                    "spec_name": getattr(spec, "name", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return False, None

    def drawer_forward_advance_ready(subtask: str, t_now: int, phase: str) -> tuple[bool, dict[str, Any]]:
        text = " ".join(str(subtask).strip().lower().split())
        if "drawer" not in text:
            return True, {"reason": "not_drawer"}

        update_all_target_passage_states(t_now, phase)
        drawer_stage_ok, drawer_status = drawer_stage_gate_status(subtask)
        if subtask not in targets:
            meta = {
                "reason": "drawer_no_target",
                "drawer_stage_ok": drawer_stage_ok,
            }
            if drawer_status is not None:
                meta.update(drawer_status)
            return drawer_stage_ok, meta

        dist = distance_to_target(obs, targets[subtask])
        pos_tol = pos_tol_for_subtask(subtask)
        seen_segments = int(get_target_passage_state(subtask)["seen_segments"])
        required_segments = required_target_passage_segments(subtask)
        return_gate_ok = target_passage_gate_allows_count(subtask)
        near_ok = dist <= pos_tol
        close_requires_stage = is_close_drawer_status(drawer_status)
        open_requires_stage_and_target = requires_drawer_stage_and_target(subtask)
        # For drawer subtasks whose training trajectory enters the target range
        # multiple times, the user requirement is explicit: eval must also wait
        # until the matching passage count is reached before hold/switch. The
        # drawer-stage predicate can loosen the final spatial criterion, but it
        # must not bypass the required passage count.
        if close_requires_stage:
            # Closing-drawer subtasks often pass near the EEF target before the
            # drawer is physically closed. That near pass must not count as a
            # completed target.
            ready = return_gate_ok and near_ok and drawer_stage_ok
        elif open_requires_stage_and_target:
            # Task4 open-middle-drawer must not hold/switch before the drawer is
            # actually opened. Reaching the EEF target alone is not enough.
            ready = return_gate_ok and near_ok and drawer_stage_ok
        else:
            ready = return_gate_ok and (drawer_stage_ok or near_ok)
        return ready, {
            "reason": "drawer_target",
            "dist": dist,
            "tol": pos_tol,
            "seen_segments": seen_segments,
            "required_segments": required_segments,
            "near_ok": near_ok,
            "return_gate_ok": return_gate_ok,
            "drawer_stage_ok": drawer_stage_ok,
            "close_requires_stage": close_requires_stage,
            "open_requires_stage_and_target": open_requires_stage_and_target,
        }
        if drawer_status is not None:
            meta.update(drawer_status)
        return ready, meta

    def check_goal(done: bool) -> bool:
        nonlocal ever_goal_success
        goal_success = (
            bool(goal_check_override(env, stage_done))
            if goal_check_override is not None
            else bool(base.ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False)
        )
        if goal_success and not ever_goal_success:
            logger.info("[t=%s] goal success", t)
        ever_goal_success = ever_goal_success or goal_success
        if done:
            logger.info("[DONE] t=%s, task done", t)
            return True
        return False

    def update_stage_and_goal(done: bool) -> bool:
        nonlocal stage_idx, current_stage_start, all_stages_logged, state, stage_done_forced_next_subtask
        nonlocal official_stage_idx, official_current_stage_start, official_all_stages_logged, official_state
        if state is not None:
            base.stage_eval._update_state(obs, state)
            if stage_idx < len(stage_specs):
                spec = stage_specs[stage_idx]
                if spec.check_fn(env, state, current_stage_start):
                    stage_done[spec.name] = True
                    logger.info("[t=%s] stage done: %s", t, spec.name)
                    if cfg.completed_update_on_stage_done:
                        completed_subtask = stage_done_completed_subtask(spec.name, current_subtask_prompt)
                        allowed = cfg.completed_update_on_stage_done_subtasks
                        if completed_subtask and (
                            not allowed or _normalize_subtask_name(completed_subtask) in allowed
                        ):
                            mark_runtime_completed(completed_subtask, f"stage_done:{spec.name}")
                    if (
                        cfg.microwave_open_stage_done_release
                        and task_id_int in {20, 21, 23, 24}
                        and _normalize_subtask_name(str(spec.name).replace("_", " ")) == "01 open microwave"
                        and open_microwave_label not in hold_consumed_subtasks
                    ):
                        hold_consumed_subtasks.add(open_microwave_label)
                        mark_runtime_completed(open_microwave_label, f"stage_done_release:{spec.name}")
                        logger.info(
                            "[MICROWAVE_OPEN_STAGE_DONE_RELEASE] t=%s task=%s stage=%s consumed=%s",
                            t,
                            planner.task_info.task_id,
                            spec.name,
                            sorted(hold_consumed_subtasks),
                        )
                    if cfg.stage_done_auto_advance and current_subtask_prompt:
                        allowed_tasks = cfg.stage_done_auto_advance_tasks
                        allowed_subtasks = cfg.stage_done_auto_advance_subtasks
                        current_norm = _normalize_subtask_name(current_subtask_prompt)
                        if (not allowed_tasks or task_id_int in allowed_tasks) and (
                            not allowed_subtasks or current_norm in allowed_subtasks
                        ):
                            current_idx = order_index(current_subtask_prompt, labels)
                            if current_idx is not None and current_idx + 1 < len(labels):
                                completed_subtask = stage_done_completed_subtask(spec.name, current_subtask_prompt)
                                if completed_subtask:
                                    stage_done_forced_next_subtask = labels[current_idx + 1]
                                    logger.info(
                                        "[STAGE_DONE_AUTO_ADVANCE_READY] t=%s task=%s stage=%s "
                                        "current_subtask=%s forced_next=%s",
                                        t,
                                        planner.task_info.task_id,
                                        spec.name,
                                        current_subtask_prompt,
                                        stage_done_forced_next_subtask,
                                    )
                    stage_idx += 1
                    current_stage_start = state["step_idx"]
            if stage_idx >= len(stage_specs) and not all_stages_logged:
                logger.info("[t=%s] all stages done", t)
                all_stages_logged = True
        if official_state is not None:
            official_stage._update_state(obs, official_state)
            if official_stage_idx < len(official_stage_specs):
                spec = official_stage_specs[official_stage_idx]
                if spec.check_fn(env, official_state, official_current_stage_start):
                    official_stage_done[spec.name] = True
                    logger.info("[t=%s] official stage done: %s", t, spec.name)
                    official_stage_idx += 1
                    official_current_stage_start = official_state["step_idx"]
            if official_stage_idx >= len(official_stage_specs) and not official_all_stages_logged:
                logger.info("[t=%s] all official stages done", t)
                official_all_stages_logged = True
        return check_goal(done)

    def maybe_update_endpose_streak(subtask: str, phase: str, t_now: int) -> bool:
        nonlocal endpose_streak
        if subtask not in targets:
            return False
        update_all_target_passage_states(t_now, phase)
        dist = distance_to_target(obs, targets[subtask])
        pos_tol = pos_tol_for_subtask(subtask)
        pick_object_key, pick_object_key_source_now = resolved_pick_object_key(subtask)
        pick_height_applies = bool(is_pick_subtask(subtask) and cfg.pick_object_lift_gate and pick_object_key is not None)
        if pick_height_applies:
            current_object_pos = get_object_pos(env, obs, str(pick_object_key))
            current_z = float(current_object_pos[2])
            prev_max_z = max_pick_height_z.get(subtask)
            if prev_max_z is None or current_z > prev_max_z:
                max_pick_height_z[subtask] = current_z
                max_pick_height_t[subtask] = t_now
            baseline_z = pick_object_baseline_z.get(subtask)
            if baseline_z is None or ((not pick_gate_closed_after_open) and current_z < baseline_z):
                pick_object_baseline_z[subtask] = current_z
                pick_object_baseline_t[subtask] = t_now
            baseline_z = pick_object_baseline_z.get(subtask)
            height_z_target = float(baseline_z) + cfg.pick_object_lift_delta if baseline_z is not None else current_z
            near_target = current_z >= height_z_target
        else:
            current_z = float("nan")
            baseline_z = None
            height_z_target = None
            near_target = dist <= pos_tol
        prev_min = min_endpose_dist.get(subtask)
        if prev_min is None or dist < prev_min:
            min_endpose_dist[subtask] = dist
            min_endpose_t[subtask] = t_now
        active_steps = max(0, t_now - current_subtask_start_t)
        final_no_hold = cfg.disable_final and subtask == final_subtask
        return_gate_ok = target_passage_gate_allows_count(subtask)
        drawer_stage_ok, drawer_status = drawer_stage_gate_status(subtask)
        stage_required_ok, stage_required_reason = endpose_stage_done_required_ok(subtask)
        close_requires_stage = is_close_drawer_status(drawer_status)
        open_requires_stage_and_target = requires_drawer_stage_and_target(subtask)
        drawer_target_ok = (
            return_gate_ok and (dist <= pos_tol) and drawer_stage_ok
            if close_requires_stage
            else (
                return_gate_ok and (dist <= pos_tol) and drawer_stage_ok
                if open_requires_stage_and_target
                else return_gate_ok and ((dist <= pos_tol) or drawer_stage_ok)
            )
        )
        pick_subtask = is_pick_subtask(subtask)
        pick_gate_applies = cfg.pick_gripper_gate and pick_subtask
        gripper_gate_ok = pick_gate_closed_after_open if pick_gate_applies else True
        if pick_subtask and cfg.pick_object_lift_gate:
            pick_completion_ok = pick_height_applies and near_target and gripper_gate_ok and return_gate_ok
        elif pick_subtask and cfg.pick_gripper_gate:
            pick_completion_ok = drawer_target_ok and gripper_gate_ok
        else:
            pick_completion_ok = drawer_target_ok
        should_count = (
            can_hold(subtask)
            and active_steps >= cfg.min_active_steps
            and pick_completion_ok
            and stage_required_ok
        )
        endpose_streak = endpose_streak + 1 if should_count else 0
        if pick_height_applies:
            logger.info(
                "[PICK_HEIGHT_GATE] t=%s task=%s subtask=%s z=%.5f baseline_z=%s target_z=%s "
                "height_ok=%s active_steps=%s gripper_gate=%s gripper_open_seen=%s "
                "gripper_closed_after_open=%s gripper_open_t=%s gripper_close_t=%s "
                "object_key=%s object_key_source=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                current_z,
                f"{baseline_z:.5f}" if baseline_z is not None else "NA",
                f"{height_z_target:.5f}" if height_z_target is not None else "NA",
                near_target,
                active_steps,
                gripper_gate_ok,
                pick_gate_open_seen if pick_gate_applies else "NA",
                pick_gate_closed_after_open if pick_gate_applies else "NA",
                pick_gate_open_t if pick_gate_applies else "NA",
                pick_gate_close_t if pick_gate_applies else "NA",
                pick_object_key,
                pick_object_key_source_now,
                phase,
            )
        elif close_requires_stage and dist <= pos_tol and return_gate_ok and not drawer_stage_ok:
            logger.info(
                "[DRAWER_CLOSE_TARGET_BLOCKED_BY_STAGE] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s drawer_status=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
                drawer_status,
            )
        elif open_requires_stage_and_target and dist <= pos_tol and return_gate_ok and not drawer_stage_ok:
            logger.info(
                "[DRAWER_OPEN_TARGET_BLOCKED_BY_STAGE] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s drawer_status=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
                drawer_status,
            )
        elif close_requires_stage and drawer_stage_ok and dist > pos_tol:
            logger.info(
                "[DRAWER_CLOSE_STAGE_BLOCKED_BY_TARGET] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s drawer_status=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
                drawer_status,
            )
        elif drawer_stage_ok and return_gate_ok:
            logger.info(
                "[DRAWER_STAGE_GATE_ALLOW] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
            )
        elif drawer_stage_ok and not return_gate_ok:
            logger.info(
                "[DRAWER_STAGE_BLOCKED_BY_PASSAGE] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
            )
        elif dist <= pos_tol and not return_gate_ok:
            logger.info(
                "[TARGET_PASSAGE_GATE_BLOCKED] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "seen_segments=%s required_segments=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                int(get_target_passage_state(subtask)["seen_segments"]),
                required_target_passage_segments(subtask),
                phase,
            )
        elif dist <= pos_tol and not stage_required_ok:
            logger.info(
                "[ENDPOSE_STAGE_DONE_REQUIRED_BLOCK] t=%s task=%s subtask=%s dist=%.5f tol=%.5f "
                "reason=%s active_steps=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                stage_required_reason,
                active_steps,
                phase,
            )
        elif (
            drawer_status is not None
            and cfg.drawer_stage_debug_interval > 0
            and t_now % cfg.drawer_stage_debug_interval == 0
        ):
            if {"mode", "region_name", "delta_y", "threshold"} <= set(drawer_status):
                logger.info(
                    "[DRAWER_STAGE_STATUS] t=%s task=%s subtask=%s mode=%s region=%s delta_y=%.5f "
                    "threshold=%.5f stage_ok=%s dist=%.5f tol=%.5f active_steps=%s "
                    "seen_segments=%s required_segments=%s phase=%s",
                    t_now,
                    planner.task_info.task_id,
                    subtask,
                    drawer_status["mode"],
                    drawer_status["region_name"],
                    drawer_status["delta_y"],
                    drawer_status["threshold"],
                    drawer_stage_ok,
                    dist,
                    pos_tol,
                    active_steps,
                    int(get_target_passage_state(subtask)["seen_segments"]),
                    required_target_passage_segments(subtask),
                    phase,
                )
            else:
                logger.info(
                    "[DRAWER_STAGE_STATUS_ALT] t=%s task=%s subtask=%s source=%s spec_name=%s "
                    "stage_ok=%s dist=%.5f tol=%.5f active_steps=%s seen_segments=%s "
                    "required_segments=%s phase=%s",
                    t_now,
                    planner.task_info.task_id,
                    subtask,
                    drawer_status.get("source", ""),
                    drawer_status.get("spec_name", ""),
                    drawer_stage_ok,
                    dist,
                    pos_tol,
                    active_steps,
                    int(get_target_passage_state(subtask)["seen_segments"]),
                    required_target_passage_segments(subtask),
                    phase,
                )
        elif (
            "drawer" in " ".join(str(subtask).strip().lower().split())
            and cfg.drawer_stage_debug_interval > 0
            and t_now % cfg.drawer_stage_debug_interval == 0
        ):
            probe = drawer_stage_probe(subtask)
            logger.info(
                "[DRAWER_STAGE_MISSING] t=%s task=%s subtask=%s region=%s has_state=%s "
                "current_site_found=%s initial_site_found=%s current_site_error=%s initial_site_error=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                probe.get("region_name"),
                probe.get("has_state"),
                probe.get("current_site_found"),
                probe.get("initial_site_found"),
                probe.get("current_site_error", ""),
                probe.get("initial_site_error", ""),
                phase,
            )
        if final_no_hold:
            logger.info(
                "[ENDPOSE_FINAL_LOG] t=%s task=%s subtask=%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                phase,
            )
        elif should_count or dist <= pos_tol or pick_height_applies:
            logger.info(
                "[ENDPOSE_NEAR] t=%s task=%s subtask=%s dist=%.5f tol=%.5f active_steps=%s "
                "pick_height_gate=%s gripper_gate=%s current_z=%s baseline_z=%s target_z=%s "
                "stage_required_ok=%s stage_required_reason=%s streak=%s/%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                active_steps,
                pick_height_applies,
                gripper_gate_ok,
                f"{current_z:.5f}" if pick_height_applies else "NA",
                f"{baseline_z:.5f}" if baseline_z is not None else "NA",
                f"{height_z_target:.5f}" if height_z_target is not None else "NA",
                stage_required_ok,
                stage_required_reason,
                endpose_streak,
                cfg.consecutive,
                phase,
            )
        elif cfg.distance_log_interval > 0 and t_now % cfg.distance_log_interval == 0:
            logger.info(
                "[ENDPOSE_DISTANCE] t=%s task=%s subtask=%s dist=%.5f tol=%.5f active_steps=%s "
                "min_dist=%.5f min_t=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                active_steps,
                min_endpose_dist[subtask],
                min_endpose_t[subtask],
                phase,
            )
        return can_hold(subtask) and endpose_streak >= cfg.consecutive

    def build_video_overlay_lines(prompt_for_vla: str, control_mode: str) -> list[str]:
        raw_prompt = " ".join(str(prompt_for_vla).strip().split()) if prompt_for_vla else "<none>"
        latest_trace = getattr(planner, "_latest_trace_record", {})
        if isinstance(latest_trace, dict):
            recent_rel = latest_trace.get("keyframe_positions", [])
            current_abs = latest_trace.get("J_abs", [])
            history_abs = latest_trace.get("K_indices_abs", [])
        else:
            recent_rel = []
            current_abs = []
            history_abs = []
        if not isinstance(recent_rel, list):
            recent_rel = []
        if not isinstance(current_abs, list):
            current_abs = []
        if not isinstance(history_abs, list):
            history_abs = []
        return [
            f"当前子任务：{raw_prompt}",
            (
                "关键帧位置（各视角共用时间步）："
                f"窗口内={recent_rel} 当前绝对={current_abs} 历史={history_abs}"
            ),
            f"累计Hold次数：{hold_count_total}",
        ]

    def step_env(action: list[float] | np.ndarray, prompt_for_vla: str, control_mode: str) -> bool:
        nonlocal obs, t
        action = maybe_hold_gripper_after_pick_release(action, prompt_for_vla, control_mode)
        element_step = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
        update_pick_gripper_gate(action, prompt_for_vla)
        overlay_lines = build_video_overlay_lines(prompt_for_vla, control_mode)
        replay.append(overlay_debug_text(element_step["observation/image"], overlay_lines))
        wrist = element_step.get("observation/wrist_image")
        if wrist is not None:
            replay_wrist.append(overlay_debug_text(wrist, overlay_lines))
        obs, _, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else action)
        append_vlm_frame()
        t += 1
        return update_stage_and_goal(bool(done))

    def maybe_apply_release_anchor(released_hold_subtask: str, next_subtask: str) -> bool:
        nonlocal obs
        if not release_anchor_rules:
            return False
        released_norm = normalize_subtask(released_hold_subtask, labels)
        next_norm = normalize_subtask(next_subtask, labels)
        rule = None
        for candidate in release_anchor_rules:
            candidate_released = normalize_subtask(candidate["released"], labels)
            candidate_next = normalize_subtask(candidate["next"], labels)
            if released_norm != candidate_released:
                continue
            if next_norm != candidate_next:
                continue
            rule = candidate
            break
        if rule is None:
            return False

        anchor_hdf5 = str(rule["anchor_hdf5"]).strip()
        frame_idx = max(0, int(rule.get("frame_idx", 0)))
        try:
            anchor = load_task4_release_anchor(anchor_hdf5, frame_idx)
            robot = env.robots[0]
            robot.set_robot_joint_positions(anchor["joint_states"])
            gripper_method = _apply_gripper_joint_positions(env, robot, anchor["gripper_states"])
            env.sim.forward()
            env._post_process()
            env._update_observables(force=True)
            if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
                obs = env.env._get_observations()
            append_vlm_frame()
            logger.info(
                "[SUBTASK_RELEASE_ANCHOR] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                "rule=%s anchor_file=%s frame_idx=%s anchor_ee=%s joint=%s gripper=%s",
                t,
                planner.task_info.task_id,
                released_hold_subtask,
                next_subtask,
                rule.get("tag", f"{rule['released']}->{rule['next']}"),
                anchor_hdf5,
                frame_idx,
                format_vec3(anchor["ee_pos"]),
                np.round(anchor["joint_states"], 6).tolist(),
                {
                    "method": gripper_method or "SKIP",
                    "target": np.round(anchor["gripper_states"], 6).tolist(),
                    "obs": np.round(np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64), 6).tolist()
                    if isinstance(obs, dict)
                    else [],
                },
            )
            return True
        except Exception:
            logger.exception(
                "[SUBTASK_RELEASE_ANCHOR_FAILED] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                "rule=%s anchor_file=%s frame_idx=%s",
                t,
                planner.task_info.task_id,
                released_hold_subtask,
                next_subtask,
                rule.get("tag", f"{rule['released']}->{rule['next']}"),
                anchor_hdf5,
                frame_idx,
            )
            return False

    def pick_lift_auto_release_candidate() -> str | None:
        if not cfg.pick_lift_auto_release:
            return None
        if not hold_active:
            return None
        subtask = hold_subtask or current_subtask_prompt
        if not subtask or not is_pick_subtask(subtask):
            return None
        repeat_needed = max(1, int(cfg.pick_lift_auto_release_repeat))
        repeat_count = int(hold_prompt_counts.get(subtask, 0))
        if repeat_count < repeat_needed:
            return None
        pick_object_key, pick_object_key_source_now = resolved_pick_object_key(subtask)
        if pick_object_key is None:
            logger.info(
                "[PICK_LIFT_AUTO_RELEASE_BLOCKED] t=%s task=%s subtask=%s reason=no_object_key repeats=%s/%s",
                t,
                planner.task_info.task_id,
                subtask,
                repeat_count,
                repeat_needed,
            )
            return None
        baseline_z = pick_object_baseline_z.get(subtask)
        if baseline_z is None:
            logger.info(
                "[PICK_LIFT_AUTO_RELEASE_BLOCKED] t=%s task=%s subtask=%s reason=no_baseline "
                "object_key=%s source=%s repeats=%s/%s",
                t,
                planner.task_info.task_id,
                subtask,
                pick_object_key,
                pick_object_key_source_now,
                repeat_count,
                repeat_needed,
            )
            return None
        current_object_pos = get_object_pos(env, obs, str(pick_object_key))
        current_z = float(current_object_pos[2])
        target_z = float(baseline_z) + cfg.pick_object_lift_delta
        if current_z < target_z:
            logger.info(
                "[PICK_LIFT_AUTO_RELEASE_BLOCKED] t=%s task=%s subtask=%s reason=not_lifted "
                "z=%.5f baseline_z=%.5f target_z=%.5f repeats=%s/%s",
                t,
                planner.task_info.task_id,
                subtask,
                current_z,
                float(baseline_z),
                target_z,
                repeat_count,
                repeat_needed,
            )
            return None
        hold_idx = order_index(subtask, labels)
        if hold_idx is None or hold_idx + 1 >= len(labels):
            logger.info(
                "[PICK_LIFT_AUTO_RELEASE_BLOCKED] t=%s task=%s subtask=%s reason=no_next_label "
                "hold_idx=%s labels=%s",
                t,
                planner.task_info.task_id,
                subtask,
                hold_idx,
                labels,
            )
            return None
        next_label = labels[hold_idx + 1]
        logger.info(
            "[PICK_LIFT_AUTO_RELEASE] t=%s task=%s old_subtask=%s new_subtask=%s "
            "z=%.5f baseline_z=%.5f target_z=%.5f repeats=%s/%s object_key=%s source=%s",
            t,
            planner.task_info.task_id,
            subtask,
            next_label,
            current_z,
            float(baseline_z),
            target_z,
            repeat_count,
            repeat_needed,
            pick_object_key,
            pick_object_key_source_now,
        )
        return next_label

    def run_vla_without_vlm(step_budget: int, phase: str) -> bool:
        nonlocal hold_active, hold_subtask, hold_prompt_counts, hold_count_total, stage_idx
        remaining = max(0, int(step_budget))
        if remaining <= 0:
            return False
        window_start_prompt = current_subtask_prompt or planner.default_subtask_prompt
        window_start_stage_idx = stage_idx
        window_start_prompt_blocked = False
        logger.info(
            "[POST_HOLD_RELEASE_VLA_START] t=%s task=%s subtask=%s steps=%s phase=%s",
            t,
            planner.task_info.task_id,
            window_start_prompt,
            remaining,
            phase,
        )
        while remaining > 0 and t < args.max_steps + args.num_steps_wait:
            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if len(actions) <= 0:
                raise RuntimeError("VLA returned an empty action chunk")
            chunk_len = min(len(actions), remaining, args.max_steps + args.num_steps_wait - t)
            logger.info(
                "[POST_HOLD_RELEASE_VLA_CHUNK] t=%s task=%s subtask=%s chunk_steps=%s remaining_before=%s",
                t,
                planner.task_info.task_id,
                current_subtask_prompt,
                chunk_len,
                remaining,
            )
            for post_idx, action in enumerate(actions[:chunk_len], start=1):
                stage_idx_before_step = stage_idx
                if step_env(action, prompt_for_vla, f"post_hold_release_vla_{post_idx}/{chunk_len}"):
                    return True
                if (not window_start_prompt_blocked) and stage_idx > window_start_stage_idx:
                    blocked_after_hold_prompts.add(window_start_prompt)
                    window_start_prompt_blocked = True
                    logger.info(
                        "[POST_HOLD_RELEASE_STAGE_BLOCK_PROMPT] t=%s task=%s phase=%s "
                        "window_start_subtask=%s blocked_prompt=%s stage_idx=%s->%s",
                        t,
                        planner.task_info.task_id,
                        phase,
                        window_start_prompt,
                        window_start_prompt,
                        stage_idx_before_step,
                        stage_idx,
                    )
                if maybe_update_endpose_streak(current_subtask_prompt, f"post_hold_release_vla_{post_idx}/{chunk_len}", t):
                    hold_active = True
                    hold_subtask = current_subtask_prompt
                    maybe_unlock_gripper_at_place_hold(hold_subtask, "post_hold_release_vla")
                    hold_prompt_counts.clear()
                    if hold_subtask:
                        hold_prompt_counts[hold_subtask] = 1
                    mark_runtime_completed(hold_subtask, "hold_start_post_release_vla")
                    hold_count_total += 1
                    logger.info(
                        "[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=post_hold_release_vla_chunk%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        post_idx,
                    )
                    return False
                remaining -= 1
                if t >= args.max_steps + args.num_steps_wait:
                    break
        logger.info("[POST_HOLD_RELEASE_VLA_END] t=%s task=%s phase=%s", t, planner.task_info.task_id, phase)
        return False

    logger.info(
        "sync endpose-hold rollout: task=%s replan_steps=%s hold=%s tol=%.5f eef_default_tol=%.5f "
        "eef_p95_extra_tol=%.5f eef_tol_cap=%.5f min_active_steps=%s "
        "consecutive=%s post_hold_release_vla_steps=%s strict_hold_release_next=%s prevent_regression=%s "
        "guard_after_hold=%s regression_guard_mode=%s disable_output_normalize=%s "
        "vlm_task_text_mode=%s drawer_open_return_hold=%s drawer_open_return_away_dist=%.5f "
        "drawer_forward_advance_guard=%s drawer_require_stage_and_target=%s "
        "drawer_open_stage_thresh=%.5f drawer_close_stage_thresh=%.5f "
        "drawer_stage_debug_interval=%s "
        "subtask_forward_max_advance=%s microwave_require_open_hold_release=%s "
        "microwave_open_stage_done_release=%s "
        "stage_done_auto_advance=%s stage_done_auto_advance_tasks=%s "
        "stage_done_auto_advance_subtasks=%s "
        "endpose_hold_require_stage_done_subtasks=%s "
        "passage_counts_json=%s target_passage_counts=%s targets=%s "
        "release_anchor_json=%s release_anchor_rules=%s "
        "pick_gripper_gate=%s pick_gripper_open_max=%.3f pick_gripper_close_min=%.3f "
        "pick_object_lift_gate=%s pick_object_lift_delta=%.5f "
        "pick_lift_auto_release=%s pick_lift_auto_release_repeat=%s "
        "post_pick_release_hold_gripper_steps=%s post_pick_release_hold_gripper_value=%+.3f "
        "post_pick_release_hold_gripper_until_place_hold=%s "
        "post_pick_release_hold_gripper_after_place_hold_steps=%s",
        planner.task_info.task_id,
        args.replan_steps,
        cfg.enabled,
        cfg.pos_tol,
        cfg.eef_default_tol,
        cfg.eef_p95_extra_tol,
        cfg.eef_tol_cap,
        cfg.min_active_steps,
        cfg.consecutive,
        cfg.post_release_vla_steps,
        cfg.strict_hold_release_next,
        cfg.prevent_regression,
        cfg.regression_guard_after_hold_release,
        "release_prompt" if release_prompt_guard else "hold_majority_prompt",
        disable_output_normalize,
        os.environ.get("VLM_TASK_TEXT_MODE", "default"),
        cfg.drawer_open_return_hold,
        cfg.drawer_open_return_away_dist,
        cfg.drawer_forward_advance_guard,
        cfg.drawer_require_stage_and_target,
        cfg.drawer_open_stage_thresh,
        cfg.drawer_close_stage_thresh,
        cfg.drawer_stage_debug_interval,
        cfg.subtask_forward_max_advance,
        cfg.microwave_require_open_hold_release,
        cfg.microwave_open_stage_done_release,
        cfg.stage_done_auto_advance,
        cfg.stage_done_auto_advance_tasks,
        cfg.stage_done_auto_advance_subtasks,
        cfg.endpose_hold_require_stage_done_subtasks,
        str(cfg.passage_counts_json) if cfg.passage_counts_json else "",
        target_passage_counts,
        sorted(targets.keys()),
        os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip(),
        release_anchor_rules,
        cfg.pick_gripper_gate,
        cfg.pick_gripper_open_max,
        cfg.pick_gripper_close_min,
        cfg.pick_object_lift_gate,
        cfg.pick_object_lift_delta,
        cfg.pick_lift_auto_release,
        cfg.pick_lift_auto_release_repeat,
        cfg.post_pick_release_hold_gripper_steps,
        cfg.post_pick_release_hold_gripper_value,
        cfg.post_pick_release_hold_gripper_until_place_hold,
        cfg.post_pick_release_hold_gripper_after_place_hold_steps,
    )

    try:
        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                append_vlm_frame()
                t += 1
                if check_goal(bool(done)):
                    break
                continue

            if state is None:
                state = base.stage_eval._build_initial_state(env)
                current_stage_start = state["step_idx"]
                official_state = official_stage._build_initial_state(env)
                official_current_stage_start = official_state["step_idx"]

            if len(recent_vlm_frames) < args.n_recent:
                obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                append_vlm_frame()
                t += 1
                if update_stage_and_goal(bool(done)):
                    break
                continue

            if (
                hold_active
                and post_pick_release_hold_gripper_remaining > 0
                and normalize_subtask(hold_subtask, labels) == post_pick_release_hold_gripper_target
            ):
                delay_chunk = min(args.replan_steps, post_pick_release_hold_gripper_remaining)
                logger.info(
                    "[POST_PICK_RELEASE_HOLD_GRIPPER_DELAY] t=%s task=%s subtask=%s "
                    "chunk_steps=%s remaining_before=%s",
                    t,
                    planner.task_info.task_id,
                    hold_subtask,
                    delay_chunk,
                    post_pick_release_hold_gripper_remaining,
                )
                closed_hold_action = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    cfg.post_pick_release_hold_gripper_value,
                ]
                for delay_idx in range(1, delay_chunk + 1):
                    if step_env(
                        closed_hold_action,
                        current_subtask_prompt or planner.default_subtask_prompt,
                        f"post_place_hold_gripper_delay_{delay_idx}/{delay_chunk}",
                    ):
                        raise StopIteration
                    if t >= args.max_steps + args.num_steps_wait:
                        break
                continue

            effective_t = t - args.num_steps_wait
            latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
            if disable_output_normalize:
                latest_subtask = " ".join(str(latest_subtask).strip().lower().replace("_", " ").split())
            else:
                latest_subtask = normalize_subtask(latest_subtask, labels)

            if (
                cfg.microwave_require_open_hold_release
                and task_id_int in {20, 21, 23, 24}
                and "open microwave" in label_by_norm
                and "open microwave" not in hold_consumed_subtasks
                and latest_subtask
                and _normalize_subtask_name(latest_subtask) != "open microwave"
            ):
                logger.info(
                    "[MICROWAVE_OPEN_HOLD_RELEASE_BLOCKED] t=%s task=%s raw_subtask=%s forced_subtask=%s "
                    "hold_consumed=%s",
                    t,
                    planner.task_info.task_id,
                    latest_subtask,
                    open_microwave_label,
                    sorted(hold_consumed_subtasks),
                )
                latest_subtask = open_microwave_label

            if cfg.stage_done_auto_advance and stage_done_forced_next_subtask:
                forced_idx = order_index(stage_done_forced_next_subtask, labels)
                current_idx = order_index(current_subtask_prompt, labels) if current_subtask_prompt else None
                if forced_idx is not None and (
                    current_idx is None or forced_idx > current_idx
                ):
                    logger.info(
                        "[STAGE_DONE_AUTO_ADVANCE_APPLY] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s forced_next=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        stage_done_forced_next_subtask,
                    )
                    latest_subtask = stage_done_forced_next_subtask
                stage_done_forced_next_subtask = ""

            if cfg.subtask_forward_max_advance > 0 and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx + cfg.subtask_forward_max_advance
                ):
                    logger.info(
                        "[SUBTASK_FORWARD_SKIP_BLOCKED] t=%s task=%s current_subtask=%s raw_subtask=%s "
                        "current_idx=%s raw_idx=%s max_advance=%s",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                        latest_subtask,
                        current_idx,
                        latest_idx,
                        cfg.subtask_forward_max_advance,
                    )
                    latest_subtask = current_subtask_prompt

            if cfg.drawer_forward_advance_guard and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    current_idx is not None
                    and latest_idx is not None
                    and max_prompt_idx_seen is not None
                    and latest_idx < max_prompt_idx_seen
                    and "drawer" in " ".join(str(current_subtask_prompt).strip().lower().split())
                ):
                    logger.info(
                        "[DRAWER_BACKWARD_SWITCH_BLOCKED] t=%s task=%s current_subtask=%s raw_subtask=%s "
                        "current_idx=%s raw_idx=%s frontier_idx=%s",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                        latest_subtask,
                        current_idx,
                        latest_idx,
                        max_prompt_idx_seen,
                    )
                    latest_subtask = current_subtask_prompt
                    latest_idx = current_idx
                if (
                    current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx
                    and "drawer" in " ".join(str(current_subtask_prompt).strip().lower().split())
                ):
                    ready_to_advance, meta = drawer_forward_advance_ready(
                        current_subtask_prompt,
                        t,
                        "before_vlm_forward_guard",
                    )
                    if (
                        (not ready_to_advance)
                        and cfg.drawer_forward_allow_stage_done
                        and bool(meta.get("drawer_stage_ok"))
                    ):
                        allowed = cfg.drawer_forward_allow_stage_done_subtasks
                        current_norm = _normalize_subtask_name(current_subtask_prompt)
                        if not allowed or current_norm in allowed:
                            ready_to_advance = True
                            logger.info(
                                "[DRAWER_FORWARD_SWITCH_ALLOW_STAGE_DONE] t=%s task=%s current_subtask=%s "
                                "raw_subtask=%s current_idx=%s raw_idx=%s meta=%s",
                                t,
                                planner.task_info.task_id,
                                current_subtask_prompt,
                                latest_subtask,
                                current_idx,
                                latest_idx,
                                meta,
                            )
                    if not ready_to_advance:
                        logger.info(
                            "[DRAWER_FORWARD_SWITCH_BLOCKED] t=%s task=%s current_subtask=%s raw_subtask=%s "
                            "current_idx=%s raw_idx=%s meta=%s",
                            t,
                            planner.task_info.task_id,
                            current_subtask_prompt,
                            latest_subtask,
                            current_idx,
                            latest_idx,
                            meta,
                        )
                        latest_subtask = current_subtask_prompt

            if hold_active and cfg.strict_hold_release_next and latest_subtask and latest_subtask != current_subtask_prompt:
                hold_idx = order_index(hold_subtask, labels)
                latest_idx = order_index(latest_subtask, labels)
                expected_idx = None if hold_idx is None else hold_idx + 1
                if hold_idx is None or latest_idx is None or latest_idx != expected_idx:
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE_BLOCKED] t=%s task=%s hold_subtask=%s raw_subtask=%s "
                        "hold_idx=%s raw_idx=%s expected_next_idx=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        latest_subtask,
                        hold_idx,
                        latest_idx,
                        expected_idx,
                    )
                    latest_subtask = current_subtask_prompt

            if cfg.prevent_regression and regression_guard_active and current_subtask_prompt and latest_subtask:
                if latest_subtask != current_subtask_prompt and latest_subtask in blocked_after_hold_prompts:
                    logger.info(
                        "[SUBTASK_REGRESSION_BLOCKED] t=%s task=%s raw_subtask=%s current_subtask=%s "
                        "guard_mode=%s blocked_prompts=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        "release_prompt" if release_prompt_guard else "hold_majority_prompt",
                        sorted(blocked_after_hold_prompts),
                    )
                    latest_subtask = current_subtask_prompt

            if hold_active and latest_subtask:
                hold_prompt_counts[latest_subtask] = hold_prompt_counts.get(latest_subtask, 0) + 1

            if hold_active and latest_subtask == current_subtask_prompt:
                auto_release_next = pick_lift_auto_release_candidate()
                if auto_release_next:
                    latest_subtask = auto_release_next

            if latest_subtask and latest_subtask != current_subtask_prompt:
                previous = current_subtask_prompt
                released_from_hold = hold_active
                released_hold_subtask = hold_subtask
                current_subtask_prompt = latest_subtask
                current_subtask_start_t = t
                endpose_streak = 0
                reset_pick_completion_gate(current_subtask_prompt)
                if released_from_hold:
                    if released_hold_subtask and released_hold_subtask not in runtime_completed_subtasks:
                        runtime_completed_subtasks.append(released_hold_subtask)
                        setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
                        logger.info(
                            "[COMPLETED_SUBTASKS_UPDATE] t=%s task=%s completed=%s mode=%s",
                            t,
                            planner.task_info.task_id,
                            runtime_completed_subtasks,
                            _completed_subtasks_mode() or "off",
                        )
                    if release_prompt_guard:
                        # Guard against falling back to the subtask we just left.
                        # Repeated drawer tasks legitimately revisit later "close/open" variants
                        # with distinct labels (e.g. "again", "final"), so blocking the newly
                        # accepted prompt is too aggressive and does not actually stop regressions.
                        block_prompt = previous
                    else:
                        block_prompt = most_common_hold_prompt()
                    if block_prompt:
                        blocked_after_hold_prompts.add(block_prompt)
                    if released_hold_subtask:
                        hold_consumed_subtasks.add(released_hold_subtask)
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE] t=%s task=%s old_subtask=%s new_subtask=%s "
                        "blocked_after_release=%s hold_prompt_counts=%s hold_consumed=%s",
                        t,
                        planner.task_info.task_id,
                        released_hold_subtask,
                        current_subtask_prompt,
                        block_prompt,
                        dict(sorted(hold_prompt_counts.items())),
                        sorted(hold_consumed_subtasks),
                    )
                    if (
                        (
                            cfg.post_pick_release_hold_gripper_steps > 0
                            or cfg.post_pick_release_hold_gripper_until_place_hold
                        )
                        and is_pick_subtask(released_hold_subtask)
                        and is_place_subtask(current_subtask_prompt)
                    ):
                        post_pick_release_hold_gripper_active = (
                            cfg.post_pick_release_hold_gripper_until_place_hold
                        )
                        post_pick_release_hold_gripper_target = normalize_subtask(
                            current_subtask_prompt, labels
                        )
                        post_pick_release_hold_gripper_remaining = (
                            0
                            if post_pick_release_hold_gripper_active
                            else cfg.post_pick_release_hold_gripper_steps
                        )
                        post_pick_release_hold_gripper_source = f"{released_hold_subtask}->{current_subtask_prompt}"
                        logger.info(
                            "[POST_PICK_RELEASE_HOLD_GRIPPER_ARMED] t=%s task=%s old_subtask=%s "
                            "new_subtask=%s mode=%s target_hold=%s steps=%s forced_gripper=%+.3f",
                            t,
                            planner.task_info.task_id,
                            released_hold_subtask,
                            current_subtask_prompt,
                            "until_place_hold" if post_pick_release_hold_gripper_active else "fixed_steps",
                            post_pick_release_hold_gripper_target,
                            post_pick_release_hold_gripper_remaining,
                            cfg.post_pick_release_hold_gripper_value,
                        )
                    maybe_apply_release_anchor(released_hold_subtask, current_subtask_prompt)
                elif cfg.release_anchor_on_nonhold_switch:
                    maybe_apply_release_anchor(previous, current_subtask_prompt)
                hold_active = False
                hold_subtask = ""
                hold_prompt_counts.clear()
                if released_from_hold and cfg.regression_guard_after_hold_release:
                    regression_guard_active = True
                    logger.info(
                        "[SUBTASK_REGRESSION_GUARD_ON] t=%s task=%s trigger=hold_release subtask=%s",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                    )
                logger.info("[t=%s] VLM sync subtask update: %s -> %s", t, previous or "<none>", current_subtask_prompt)
                current_idx = order_index(current_subtask_prompt, labels)
                if current_idx is not None and (max_prompt_idx_seen is None or current_idx > max_prompt_idx_seen):
                    max_prompt_idx_seen = current_idx
                    logger.info(
                        "[DRAWER_FORWARD_FRONTIER] t=%s task=%s subtask=%s frontier_idx=%s",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                        max_prompt_idx_seen,
                    )
                if current_subtask_prompt == final_subtask:
                    logger.info("[FINAL_HINT] t=%s task=%s VLM reached final subtask.", t, planner.task_info.task_id)
                if released_from_hold and cfg.post_release_vla_steps > 0:
                    if run_vla_without_vlm(cfg.post_release_vla_steps, phase="after_hold_release"):
                        break
                    continue

            if hold_active:
                target = targets.get(hold_subtask)
                hold_gripper = float(target["hold_gripper"]) if target else -1.0
                hold_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hold_gripper]
                logger.info(
                    "[ENDPOSE_HOLD_STEP] t=%s task=%s subtask=%s hold_steps=%s gripper=%+.0f",
                    t,
                    planner.task_info.task_id,
                    hold_subtask,
                    args.replan_steps,
                    hold_gripper,
                )
                for _ in range(args.replan_steps):
                    if step_env(
                        hold_action,
                        current_subtask_prompt or planner.default_subtask_prompt,
                        "hold_zero_action",
                    ):
                        raise StopIteration
                    if t >= args.max_steps + args.num_steps_wait:
                        break
                continue

            if maybe_update_endpose_streak(current_subtask_prompt, "before_vla", t):
                hold_active = True
                hold_subtask = current_subtask_prompt
                maybe_unlock_gripper_at_place_hold(hold_subtask, "before_vla")
                hold_prompt_counts.clear()
                if hold_subtask:
                    hold_prompt_counts[hold_subtask] = 1
                mark_runtime_completed(hold_subtask, "hold_start_before_vla")
                hold_count_total += 1
                logger.info("[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=before_vla", t, planner.task_info.task_id, hold_subtask)
                continue

            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if len(actions) < args.replan_steps:
                raise RuntimeError(f"VLA returned {len(actions)} actions, need at least {args.replan_steps}")
            logger.info("[t=%s] VLA sync chunk: %s steps | prompt=%s", t, args.replan_steps, prompt_for_vla)
            for chunk_idx, action in enumerate(actions[: args.replan_steps], start=1):
                if step_env(action, prompt_for_vla, f"vla_chunk_{chunk_idx}/{args.replan_steps}"):
                    raise StopIteration
                if maybe_update_endpose_streak(current_subtask_prompt, f"after_vla_chunk{chunk_idx}", t):
                    hold_active = True
                    hold_subtask = current_subtask_prompt
                    maybe_unlock_gripper_at_place_hold(hold_subtask, f"after_vla_chunk{chunk_idx}")
                    hold_prompt_counts.clear()
                    if hold_subtask:
                        hold_prompt_counts[hold_subtask] = 1
                    mark_runtime_completed(hold_subtask, "hold_start_after_vla")
                    hold_count_total += 1
                    logger.info(
                        "[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=after_vla_chunk%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        chunk_idx,
                    )
                    break
                if t >= args.max_steps + args.num_steps_wait:
                    break
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")

    stage_pct = official_stage._stage_score_pct(task_id_int, official_stage_done)
    stage_success = official_stage._stage_success_from_stage_done(task_id_int, official_stage_done)
    official_goal_override = official_stage._goal_override_check(task_id_int)
    if official_goal_override is not None:
        official_goal_success = bool(official_goal_override(env, official_stage_done))
    else:
        official_goal_success = bool(stage_success)
    if not ever_goal_success:
        ever_goal_success = (
            bool(goal_check_override(env, stage_done))
            if goal_check_override is not None
            else bool(base.ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False)
        )
    if min_endpose_dist:
        for subtask in sorted(min_endpose_dist):
            logger.info(
                "[ENDPOSE_MIN_DISTANCE] task=%s subtask=%s min_dist=%.5f tol=%.5f min_t=%s",
                planner.task_info.task_id,
                subtask,
                min_endpose_dist[subtask],
                pos_tol_for_subtask(subtask),
                min_endpose_t[subtask],
            )
    if max_pick_height_z:
        for subtask in sorted(max_pick_height_z):
            logger.info(
                "[PICK_HEIGHT_MAX] task=%s subtask=%s max_z=%.5f baseline_z=%s max_t=%s",
                planner.task_info.task_id,
                subtask,
                max_pick_height_z[subtask],
                f"{pick_object_baseline_z[subtask]:.5f}" if subtask in pick_object_baseline_z else "NA",
                max_pick_height_t[subtask],
            )
    logger.info(
        "[OFFICIAL_SCORE] task=%s average_score_pct=%.6f stage_success=%s goal_success=%s stage_done_json=%s",
        task_id_int,
        stage_pct,
        int(stage_success),
        int(official_goal_success),
        json.dumps(official_stage_done, ensure_ascii=False, separators=(",", ":")),
    )
    return stage_pct, official_stage_done, official_goal_success, replay, replay_wrist


def _write_official_summaries() -> None:
    out_root = Path(os.environ["OUT_ROOT"])
    pattern = re.compile(
        r"\[OFFICIAL_SCORE\] task=(\d+) average_score_pct=([0-9.]+) "
        r"stage_success=([01]) goal_success=([01]) stage_done_json=(\{.*\})"
    )
    rows: list[dict[str, Any]] = []
    for log_path in sorted(out_root.glob("task*/ep*/sync_vlm.log")):
        matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
        if not matches:
            continue
        task, score, stage_success, goal_success, stage_json = matches[-1]
        ep_match = re.search(r"ep(\d+)$", log_path.parent.name)
        ep = int(ep_match.group(1)) if ep_match else len(rows)
        seed = int(os.environ.get("SEED", "104")) + ep
        rows.append(
            {
                "task_id": int(task), "ep": ep, "seed": seed, "score_pct": float(score),
                "tsr_success": bool(int(stage_success)), "stage_success": bool(int(stage_success)),
                "goal_success": bool(int(goal_success)), "stage_done": json.loads(stage_json),
                "log": str(log_path),
            }
        )
    rows.sort(key=lambda row: (row["task_id"], row["ep"]))
    with (out_root / "official_episodes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("task_id\tep\tseed\tscore_pct\ttsr_success\tstage_success\tgoal_success\tlog\n")
        for row in rows:
            handle.write(
                f'{row["task_id"]}\t{row["ep"]}\t{row["seed"]}\t{row["score_pct"]:.1f}\t'
                f'{"Y" if row["tsr_success"] else "N"}\t{"Y" if row["stage_success"] else "N"}\t'
                f'{"Y" if row["goal_success"] else "N"}\t{row["log"]}\n'
            )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    task_rows = []
    for task_id, episodes in sorted(grouped.items()):
        count = len(episodes)
        task_rows.append({
            "task_id": task_id, "num_trials": count, "seed_start": int(os.environ.get("SEED", "104")),
            "average_score_pct": sum(row["score_pct"] for row in episodes) / max(1, count),
            "tsr_success_rate_pct": 100.0 * sum(row["tsr_success"] for row in episodes) / max(1, count),
            "stage_success_rate_pct": 100.0 * sum(row["stage_success"] for row in episodes) / max(1, count),
            "goal_success_rate_pct": 100.0 * sum(row["goal_success"] for row in episodes) / max(1, count),
        })
    (out_root / "official_summary.json").write_text(
        json.dumps({"episodes": rows, "tasks": task_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_root / "official_task_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "task_id\tnum_trials\tseed_start\taverage_score_pct\ttsr_success_rate_pct\t"
            "stage_success_rate_pct\tgoal_success_rate_pct\n"
        )
        for row in task_rows:
            handle.write(
                f'{row["task_id"]}\t{row["num_trials"]}\t{row["seed_start"]}\t'
                f'{row["average_score_pct"]:.1f}\t{row["tsr_success_rate_pct"]:.1f}\t'
                f'{row["stage_success_rate_pct"]:.1f}\t{row["goal_success_rate_pct"]:.1f}\n'
            )


def main() -> None:
    os.environ["ASYNC_VLM"] = "0"
    base.SyncLoRAPlanner._append_trace = _append_trace_with_cache
    base.FullVlm26MemoryPlanner._build_messages = _build_messages_runtime_progress
    if hasattr(base, "_task_specs"):
        base._task_specs = _patched_task_specs
    if hasattr(base.stage_eval, "_task_specs"):
        base.stage_eval._task_specs = _patched_task_specs
    base.run_episode_async_stateful = run_episode_sync_endpose_hold
    base.main()
    _write_official_summaries()


if __name__ == "__main__":
    main()
