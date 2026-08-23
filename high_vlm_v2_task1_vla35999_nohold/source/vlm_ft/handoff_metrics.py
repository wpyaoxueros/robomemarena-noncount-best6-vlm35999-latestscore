"""Original-trajectory SCNA boundaries and physical valid-handoff windows.

SCNA and VHWS deliberately use different ground truth.  SCNA is anchored to
the HDF5 segment boundary.  VHWS asks whether the VLM's first switch to the
next primitive begins inside a scene-grounded interval: after the old action
has physically released/departed its interaction, and before the next action
has physically engaged its target.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from semantic_metrics import BoundaryRecord, PredictionRecord, TASK_PRIMITIVES


OBJECT_FOR_PRIMITIVE: dict[int, dict[str, str]] = {
    1: {
        "pick cookies": "cookies_1",
        "place cookies into basket": "cookies_1",
        "pick tomato sauce": "tomato_sauce_1",
        "place tomato into basket": "tomato_sauce_1",
    },
    17: {
        "pick butter": "butter_1",
        "place butter": "butter_1",
        "pick chocolate": "chocolate_pudding_1",
        "place chocolate": "chocolate_pudding_1",
    },
    22: {
        "pick tomato": "tomato_sauce_1",
        "pour first": "tomato_sauce_1",
        "pour second": "tomato_sauce_1",
        "place tomato aside": "tomato_sauce_1",
        "pick cookies": "cookies_1",
        "place cookies": "cookies_1",
    },
}


VHWS_RULES: dict[int, tuple[dict[str, str], ...]] = {
    1: (
        {"start": "pick complete (stable grasp/lift)", "end": "cookies first enter the basket region"},
        {"start": "cookies released, then EEF departs cookies", "end": "EEF first reaches tomato-sauce contact zone"},
        {"start": "pick complete (stable grasp/lift)", "end": "tomato sauce first enters the basket region"},
    ),
    17: (
        {"start": "drawer open, then EEF departs handle pose", "end": "EEF first reaches butter contact zone"},
        {"start": "butter pick complete", "end": "butter first enters the middle-drawer region"},
        {"start": "butter released, then EEF departs butter", "end": "EEF first reaches chocolate contact zone"},
        {"start": "chocolate pick complete", "end": "chocolate first enters the middle-drawer region"},
        {"start": "chocolate released, then EEF departs chocolate", "end": "middle drawer first begins closing"},
    ),
    22: (
        {"start": "tomato pick complete", "end": "first pour tilt begins"},
        {"start": "first pour cycle complete", "end": "second pour tilt begins"},
        {"start": "second pour cycle complete", "end": "tomato reaches the aside placement/contact zone"},
        {"start": "tomato released, then EEF departs tomato", "end": "microwave door first begins opening"},
        {"start": "microwave open, then EEF departs handle pose", "end": "EEF first reaches cookies contact zone"},
        {"start": "cookies pick complete", "end": "cookies first enter the microwave heating region"},
        {"start": "cookies released, then EEF departs cookies", "end": "microwave door first begins closing"},
    ),
}


@dataclass(frozen=True)
class HandoffWindow:
    task_id: int
    episode: int
    seed: int
    boundary_index: int
    old_primitive: str
    next_primitive: str
    old_activation_step: int | None
    window_start_step: int | None
    window_end_step: int | None
    start_rule: str
    end_rule: str
    start_signal: str
    end_signal: str


def trajectory_boundaries(
    task_id: int,
    episode: int,
    seed: int,
    manifest: Sequence[dict[str, Any]],
    episode_end_step: int,
    *,
    labels: Sequence[str] | None = None,
) -> list[BoundaryRecord]:
    """Build SCNA boundaries solely from original HDF5 segment endpoints."""

    labels = tuple(labels) if labels is not None else TASK_PRIMITIVES[int(task_id)]
    if len(manifest) != len(labels):
        raise ValueError(f"task{task_id}: {len(manifest)} GT segments for {len(labels)} primitives")
    rows: list[BoundaryRecord] = []
    for index in range(len(labels) - 1):
        current = manifest[index]
        nxt = manifest[index + 1]
        rows.append(
            BoundaryRecord(
                task_id=int(task_id),
                episode=int(episode),
                seed=int(seed),
                boundary_index=index,
                old_primitive=labels[index],
                next_primitive=labels[index + 1],
                activated_step=int(current["start_action_step"]),
                completion_step=int(current["end_action_step_exclusive"]),
                next_completion_step=int(nxt["end_action_step_exclusive"]),
                episode_end_step=int(episode_end_step),
                boundary_source="original_hdf5_trajectory",
            )
        )
    return rows


def _vec(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array if array.size >= 3 and np.all(np.isfinite(array[:3])) else None


def _physical_by_step(states: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row.get("physical", {}) for row in states if "step" in row}


def _first_stable_step(steps: Sequence[int], predicate: Any, stable_steps: int = 3) -> int | None:
    streak: list[int] = []
    for step in steps:
        if bool(predicate(step)):
            streak.append(step)
            if len(streak) >= stable_steps:
                return streak[-stable_steps]
        else:
            streak.clear()
    return None


def _object_row(physical: dict[str, Any], name: str) -> dict[str, Any]:
    return physical.get("objects", {}).get(name, {})


def _departure_step(
    physical: dict[int, dict[str, Any]], steps: Sequence[int], object_name: str, threshold: float = 0.10
) -> int | None:
    def departed(step: int) -> bool:
        eef = _vec(physical[step].get("robot0_eef_pos"))
        obj = _vec(_object_row(physical[step], object_name).get("position"))
        grasped = _object_row(physical[step], object_name).get("grasped")
        return eef is not None and obj is not None and grasped is not True and np.linalg.norm(eef - obj) >= threshold

    return _first_stable_step(steps, departed)


def _depart_pose_step(
    physical: dict[int, dict[str, Any]], steps: Sequence[int], origin_step: int, threshold: float = 0.08
) -> int | None:
    origin = _vec(physical.get(origin_step, {}).get("robot0_eef_pos"))
    if origin is None:
        return None
    return _first_stable_step(
        steps,
        lambda step: (
            (eef := _vec(physical[step].get("robot0_eef_pos"))) is not None
            and np.linalg.norm(eef - origin) >= threshold
        ),
    )


def _pick_engagement_step(
    physical: dict[int, dict[str, Any]], steps: Sequence[int], object_name: str, threshold: float = 0.075
) -> int | None:
    def engaged(step: int) -> bool:
        obj_row = _object_row(physical[step], object_name)
        if obj_row.get("grasped") is True:
            return True
        eef = _vec(physical[step].get("robot0_eef_pos"))
        obj = _vec(obj_row.get("position"))
        return eef is not None and obj is not None and np.linalg.norm(eef - obj) <= threshold

    return _first_stable_step(steps, engaged)


def _place_entry_step(
    physical: dict[int, dict[str, Any]],
    steps: Sequence[int],
    task_id: int,
    primitive: str,
    object_name: str,
) -> tuple[int | None, str]:
    """First stable entry using the benchmark's task-specific geometry."""

    # Old Task22 traces predate saving the heating-region site.  Its BDDL puts
    # the microwave origin in x=[-0.02, 0.02], y=[0.53, 0.57], while the XML
    # heating site is at local (0, 0.01).  The midpoint below has at most 2 cm
    # XY uncertainty and is preferable to using the final cookie pose, which
    # can sit near the front edge of the region.
    legacy_microwave_target = np.asarray([0.0, 0.56, 0.0], dtype=np.float64)

    def inside(step: int) -> bool:
        row = physical[step]
        obj = _vec(_object_row(physical[step], object_name).get("position"))
        if obj is None:
            return False
        if task_id == 1:
            target = _vec(row.get("basket_position"))
            if target is None:
                return False
            delta = np.abs(obj - target)
            return bool(delta[0] < 0.12 and delta[1] < 0.12 and -0.05 < obj[2] - target[2] < 0.20)
        if task_id == 17:
            target = _vec(row.get("middle_drawer_region_position"))
            if target is None:
                return False
            return bool(
                abs(float(obj[0] - target[0])) < 0.15
                and target[1] - 0.20 < obj[1] < target[1] + 0.10
                and abs(float(obj[2] - target[2])) < 0.10
            )
        if task_id == 22 and primitive == "place tomato aside":
            target = np.asarray([0.0, -0.2, 0.50], dtype=np.float64)
            # The benchmark completion region (0.20 m in XY and Z) already
            # overlaps the pouring pose and would collapse this handoff to a
            # zero-width window.  VHWS instead ends at the real placement
            # interaction: centered over the aside point and near table/object
            # contact height.
            return bool(np.linalg.norm(obj[:2] - target[:2]) < 0.10 and abs(float(obj[2] - target[2])) < 0.05)
        if task_id == 22 and primitive == "place cookies":
            target = _vec(row.get("microwave_heating_region_position"))
            if target is None:
                target = legacy_microwave_target
            return bool(
                target is not None
                and abs(float(obj[0] - target[0])) < 0.20
                and abs(float(obj[1] - target[1])) < 0.20
            )
        return False

    signal = {
        1: "benchmark_basket_region_stable_3_steps",
        17: "benchmark_middle_drawer_region_stable_3_steps",
    }.get(task_id)
    if task_id == 22 and primitive == "place tomato aside":
        signal = "aside_placement_contact_xy0.10_z0.05_stable_3_steps"
    if task_id == 22 and primitive == "place cookies":
        has_site = any(_vec(physical[step].get("microwave_heating_region_position")) is not None for step in steps)
        signal = (
            "benchmark_microwave_heating_region_xy0.20_stable_3_steps"
            if has_site
            else "legacy_bddl_midpoint_microwave_xy0.20_uncertainty0.02_stable_3_steps"
        )
    return _first_stable_step(steps, inside), str(signal or "unsupported_place_region")


def _device_motion_step(
    physical: dict[int, dict[str, Any]], steps: Sequence[int], key: str, origin_step: int, threshold: float
) -> int | None:
    origin = physical.get(origin_step, {}).get(key)
    if origin is None:
        return None
    return _first_stable_step(
        steps,
        lambda step: physical[step].get(key) is not None
        and abs(float(physical[step][key]) - float(origin)) >= threshold,
    )


def _completion_rows(tracker: dict[str, Any]) -> list[dict[str, Any]]:
    return list(tracker.get("primitives", []))


def build_handoff_windows(
    task_id: int,
    episode: int,
    seed: int,
    semantic_tracker: dict[str, Any],
    states: Sequence[dict[str, Any]],
) -> list[HandoffWindow]:
    """Derive task-specific physical windows from saved simulator state."""

    task_id = int(task_id)
    primitives = _completion_rows(semantic_tracker)
    labels = TASK_PRIMITIVES[task_id]
    if len(primitives) != len(labels):
        raise ValueError(f"task{task_id}: semantic tracker/primitive count mismatch")
    physical = _physical_by_step(states)
    all_steps = sorted(physical)
    rules = VHWS_RULES[task_id]
    windows: list[HandoffWindow] = []

    for index, (old_label, next_label) in enumerate(zip(labels, labels[1:])):
        old = primitives[index]
        nxt = primitives[index + 1]
        completion = old.get("completion_step")
        next_completion = nxt.get("completion_step")
        candidate_steps = [
            step for step in all_steps
            if completion is not None and step >= int(completion)
            and (next_completion is None or step <= int(next_completion))
        ]
        start: int | None = None if completion is None else int(completion)
        start_signal = "old_semantic_completion"
        old_object = OBJECT_FOR_PRIMITIVE.get(task_id, {}).get(old_label)
        if start is not None and old_label.startswith("place ") and old_object:
            start = _departure_step(physical, candidate_steps, old_object)
            start_signal = "released_and_eef_object_distance>=0.10_for_3_steps"
        elif start is not None and old_label.startswith("open "):
            start = _depart_pose_step(physical, candidate_steps, start)
            start_signal = "eef_distance_from_completion_pose>=0.08_for_3_steps"

        end: int | None = None
        end_signal = "unavailable"
        next_object = OBJECT_FOR_PRIMITIVE.get(task_id, {}).get(next_label)
        search_from = int(completion) if completion is not None else 0
        engagement_steps = [
            step for step in all_steps
            if step >= search_from and (next_completion is None or step <= int(next_completion))
        ]
        if next_label.startswith("pick ") and next_object:
            end = _pick_engagement_step(physical, engagement_steps, next_object)
            end_signal = "grasped_or_eef_object_distance<=0.075_for_3_steps"
        elif next_label.startswith("place ") and next_object:
            end, end_signal = _place_entry_step(
                physical, engagement_steps, task_id, next_label, next_object
            )
        elif next_label in {"open microwave", "close microwave"}:
            end = _device_motion_step(physical, engagement_steps, "microwave_joint_angle", search_from, 0.02)
            end_signal = "microwave_angle_changes>=0.02_for_3_steps"
        elif next_label in {"open middle drawer", "close middle drawer"}:
            end = _device_motion_step(physical, engagement_steps, "middle_drawer_displacement", search_from, 0.01)
            end_signal = "drawer_displacement_changes>=0.01_for_3_steps"
        elif next_label.startswith("pour "):
            end = _device_motion_step(physical, engagement_steps, "tomato_tilt_angle", search_from, 0.12)
            end_signal = "tomato_tilt_changes>=0.12rad_for_3_steps"

        rule = rules[index]
        windows.append(
            HandoffWindow(
                task_id=task_id,
                episode=int(episode),
                seed=int(seed),
                boundary_index=index,
                old_primitive=old_label,
                next_primitive=next_label,
                old_activation_step=old.get("activated_step"),
                window_start_step=start,
                window_end_step=end,
                start_rule=rule["start"],
                end_rule=rule["end"],
                start_signal=start_signal,
                end_signal=end_signal,
            )
        )
    return windows


def _first_run_start(predictions: Sequence[PredictionRecord], target: str, calls: int) -> PredictionRecord | None:
    for index in range(len(predictions) - calls + 1):
        if all(row.normalized_primitive == target for row in predictions[index : index + calls]):
            return predictions[index]
    return None


def evaluate_handoff_window(
    window: HandoffWindow, predictions: Sequence[PredictionRecord], stable_calls: int = 2
) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda row: row.call_index)
    active = [
        row for row in ordered
        if window.old_activation_step is None or row.input_step >= window.old_activation_step
    ]
    first_global = _first_run_start(active, window.next_primitive, 1)
    stable_global = _first_run_start(active, window.next_primitive, stable_calls)
    eligible = (
        window.window_start_step is not None
        and window.window_end_step is not None
        and window.window_start_step <= window.window_end_step
    )

    in_window = (
        []
        if not eligible
        else [
            row for row in active
            if window.window_start_step <= row.input_step <= window.window_end_step
        ]
    )
    first_valid = _first_run_start(in_window, window.next_primitive, 1)
    stable_valid = _first_run_start(in_window, window.next_primitive, stable_calls)

    return {
        **asdict(window),
        "window_valid": eligible,
        "window_width_steps": (
            window.window_end_step - window.window_start_step if eligible else None
        ),
        "num_vlm_calls_in_window": len(in_window),
        "first_global_next_call_index": None if first_global is None else first_global.call_index,
        "first_global_next_input_step": None if first_global is None else first_global.input_step,
        "first_global_stable_next_call_index": None if stable_global is None else stable_global.call_index,
        "first_global_stable_next_input_step": None if stable_global is None else stable_global.input_step,
        "first_valid_next_call_index": None if first_valid is None else first_valid.call_index,
        "first_valid_next_input_step": None if first_valid is None else first_valid.input_step,
        "first_valid_stable_next_call_index": None if stable_valid is None else stable_valid.call_index,
        "first_valid_stable_next_input_step": None if stable_valid is None else stable_valid.input_step,
        "vhws_at_1": None if not eligible or not in_window else first_valid is not None,
        "vhws_at_2": (
            None if not eligible or len(in_window) < stable_calls else stable_valid is not None
        ),
        "stable_calls_required": int(stable_calls),
    }


def summarize_vhws(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in records if row.get("window_valid")]
    observed_at_1 = [row for row in eligible if row.get("vhws_at_1") is not None]
    observed_at_2 = [row for row in eligible if row.get("vhws_at_2") is not None]
    widths = [float(row["window_width_steps"]) for row in eligible]
    return {
        "num_boundaries": len(records),
        "num_valid_windows": len(eligible),
        "window_coverage": len(eligible) / len(records) if records else None,
        "vhws_at_1": (
            sum(bool(row["vhws_at_1"]) for row in observed_at_1) / len(observed_at_1)
            if observed_at_1 else None
        ),
        "vhws_at_1_numerator": sum(bool(row["vhws_at_1"]) for row in observed_at_1),
        "vhws_at_1_denominator": len(observed_at_1),
        "vhws_at_2": (
            sum(bool(row["vhws_at_2"]) for row in observed_at_2) / len(observed_at_2)
            if observed_at_2 else None
        ),
        "vhws_at_2_numerator": sum(bool(row["vhws_at_2"]) for row in observed_at_2),
        "vhws_at_2_denominator": len(observed_at_2),
        "window_width_steps_mean": statistics.fmean(widths) if widths else None,
        "window_width_steps_median": statistics.median(widths) if widths else None,
    }
