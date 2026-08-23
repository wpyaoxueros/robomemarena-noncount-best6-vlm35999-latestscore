"""Trajectory-grounded prompt contract for the 26-task high_vlm_v2 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


SYSTEM_PROMPT = """You are an embodied-memory robot VLM planner.

Predict the low-level controller primitive assigned to the current timestep in the demonstrated trajectory segmentation. Use the latest timestep as the current state.

The input contains historical keyframe timesteps followed by exactly 5 consecutive recent timesteps. Each timestep explicitly contains an external agent-view image and a wrist-camera image. Historical keyframes are earlier than the recent window. The recent window is ordered from oldest to newest; R5 is the current timestep.

Return strict JSON only, with exactly two fields: current_primitive and keyframe_positions. current_primitive must exactly match one legal primitive listed in the user prompt. keyframe_positions is a sorted, duplicate-free, 1-indexed list of annotated keyframes inside R1..R5; use an empty list when none is present. Do not infer a semantic early switch: reproduce the demonstrated trajectory segmentation label at R5."""

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = (
    SCRIPT_DIR.parent
    / "eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation"
    / "tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"
)
CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
RECENT_STEPS = 5
MAX_HISTORY_KEYFRAMES = 8


def load_task_specs(path: Path = DEFAULT_TASK_CONFIG) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[int, dict[str, Any]] = {}
    for task in raw["tasks"]:
        order_options: list[list[dict[str, str]]] = []
        legal_primitives: list[str] = []
        stem_to_label: dict[str, str] = {}
        definitions: list[str] = []
        for order, entry in enumerate(task["primitive_order"]):
            options = entry.get("variants", [entry])
            normalized: list[dict[str, str]] = []
            for option in options:
                stem = str(option["stem"])
                label = str(option["label"])
                if stem in stem_to_label and stem_to_label[stem] != label:
                    raise ValueError(f"task{task['task_id']}: inconsistent label for {stem}")
                stem_to_label[stem] = label
                legal_primitives.append(label)
                normalized.append({"stem": stem, "label": label})
                definitions.append(f"{label} = HDF5 segment order {order} (stem: {stem}).")
            order_options.append(normalized)
        if len(set(legal_primitives)) != len(legal_primitives):
            raise ValueError(f"task{task['task_id']}: duplicate legal primitive labels")
        specs[int(task["task_id"])] = {
            "data_dir": str(task["data_dir"]),
            "objective": str(task["brief_description"]),
            "task_block": str(task["task_block"]),
            "scene": str(task.get("scene_description", "")),
            "primitives": legal_primitives,
            "order_options": order_options,
            "stem_to_label": stem_to_label,
            "definitions": definitions,
        }
    if sorted(specs) != list(range(1, 27)):
        raise ValueError(f"task config must contain ids 1..26, got {sorted(specs)}")
    return specs


TASK_SPECS = load_task_specs()


def primitive_for_segment(task_id: int, order: int, stem: str) -> str:
    spec = TASK_SPECS[int(task_id)]
    options = spec["order_options"][int(order)]
    for option in options:
        if option["stem"] == stem:
            return option["label"]
    allowed = [option["stem"] for option in options]
    raise ValueError(f"task{task_id} order{order}: stem {stem!r} not in {allowed}")


def task_text(task_id: int) -> str:
    spec = TASK_SPECS[int(task_id)]
    legal = "\n".join(f"- {item}" for item in spec["primitives"])
    definitions = "\n".join(f"- {item}" for item in spec["definitions"])
    return (
        f"Task objective: {spec['objective']}\n"
        f"Execution requirement: {spec['task_block']}\n"
        f"Scene definition: {spec['scene']}\n\n"
        f"Legal primitives (copy exactly one):\n{legal}\n\n"
        f"Trajectory-label definitions:\n{definitions}"
    )


def build_user_prompt(
    task_id: int,
    history_indices: Sequence[int],
    recent_indices: Sequence[int],
) -> str:
    if len(recent_indices) != RECENT_STEPS:
        raise ValueError(f"expected {RECENT_STEPS} recent indices")
    if len(history_indices) > MAX_HISTORY_KEYFRAMES:
        raise ValueError("history exceeds prompt contract")
    current = int(recent_indices[-1])
    lines = [task_text(task_id), "", "Image sequence (chronological within each section):"]
    if history_indices:
        lines.append(f"Historical keyframes ({len(history_indices)} timesteps; maximum 8):")
        for rank, step in enumerate(history_indices, 1):
            dt = int(step) - current
            lines.append(f"H{rank} (global_t={step}, dt={dt}) agentview_rgb: <image>")
            lines.append(f"H{rank} (global_t={step}, dt={dt}) eye_in_hand_rgb: <image>")
    else:
        lines.append("Historical keyframes: none.")
    lines.append("Recent window (exactly 5 consecutive timesteps; R5 is current):")
    for rank, step in enumerate(recent_indices, 1):
        dt = int(step) - current
        suffix = " CURRENT" if rank == RECENT_STEPS else ""
        lines.append(f"R{rank} (global_t={step}, dt={dt}) agentview_rgb{suffix}: <image>")
        lines.append(f"R{rank} (global_t={step}, dt={dt}) eye_in_hand_rgb{suffix}: <image>")
    lines.append("Predict the demonstrated trajectory-segmentation label assigned to R5.")
    return "\n".join(lines)


def build_runtime_messages(
    task_id: int,
    history_indices: Sequence[int],
    recent_indices: Sequence[int],
    history_main: Sequence[Any],
    history_wrist: Sequence[Any],
    recent_main: Sequence[Any],
    recent_wrist: Sequence[Any],
) -> list[dict[str, Any]]:
    """Build inference messages with the exact 26-task training contract."""

    images: list[Any] = []
    for main, wrist in zip(history_main, history_wrist, strict=True):
        if wrist is None:
            raise ValueError("strict dual-camera prompt requires every historical wrist image")
        images.extend((main, wrist))
    for main, wrist in zip(recent_main, recent_wrist, strict=True):
        if wrist is None:
            raise ValueError("strict dual-camera prompt requires every recent wrist image")
        images.extend((main, wrist))
    text = build_user_prompt(task_id, history_indices, recent_indices)
    if text.count("<image>") != len(images):
        raise ValueError("prompt/image count mismatch")
    parts = text.split("<image>")
    content: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if index < len(images):
            content.append({"type": "image", "image": images[index]})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
