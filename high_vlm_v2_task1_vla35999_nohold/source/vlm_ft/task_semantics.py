"""Ordered physical completion oracles for selected RoboMemArena tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from semantic_metrics import TASK_PRIMITIVES


Predicate = Callable[[Any, dict[str, Any], int], bool]


@dataclass(frozen=True)
class CompletionSpec:
    label: str
    predicate_name: str
    predicate: Predicate
    stable_steps: int


@dataclass
class PrimitiveState:
    index: int
    label: str
    predicate_name: str
    activated_step: int | None = None
    activated_state_index: int | None = None
    completion_step: int | None = None
    completion_state_index: int | None = None


def _raw_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _grasp_status(env: Any, object_name: str) -> bool | None:
    raw = _raw_env(env)
    try:
        obj = raw.get_object(object_name)
        robots = getattr(raw, "robots", None) or getattr(env, "robots", None)
        gripper = robots[0].gripper if robots else None
        if obj is None or gripper is None or not hasattr(raw, "_check_grasp"):
            return None
        return bool(raw._check_grasp(gripper, obj))  # noqa: SLF001
    except Exception:
        return None


def _current_body_pos(ref: Any, env: Any, object_name: str) -> np.ndarray | None:
    value = ref.ec._body_pos(env, object_name)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _initial_body_pos(state: dict[str, Any], object_name: str) -> np.ndarray | None:
    candidates = (object_name, f"{object_name}_main")
    for candidate in candidates:
        value = state.get("initial_body_pos", {}).get(candidate)
        if value is not None:
            return np.asarray(value, dtype=np.float64)
    return None


def _lift_delta(ref: Any, env: Any, state: dict[str, Any], object_name: str) -> float | None:
    current = _current_body_pos(ref, env, object_name)
    initial = _initial_body_pos(state, object_name)
    if current is None or initial is None:
        return None
    return float(current[2] - initial[2])


def _pick_predicate(ref: Any, object_name: str, lift_delta: float = 0.035) -> Predicate:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        del stage_start
        grasped = _grasp_status(env, object_name)
        lifted = _lift_delta(ref, env, state, object_name)
        return grasped is True or (lifted is not None and lifted > lift_delta)

    return check


def _released_and(predicate: Predicate, object_name: str) -> Predicate:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        if not predicate(env, state, stage_start):
            return False
        grasped = _grasp_status(env, object_name)
        # Old LIBERO variants do not always expose _check_grasp.  In that case
        # keep the geometric completion predicate usable and expose the missing
        # grasp signal in semantic_state.jsonl for audit.
        return grasped is not True

    return check


def _in_basket(ref: Any, object_name: str) -> Predicate:
    """Task1 basket predicate matching the benchmark's real scene geometry."""

    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        del state, stage_start
        obj = _current_body_pos(ref, env, object_name)
        basket = _current_body_pos(ref, env, "basket_1")
        if obj is None or basket is None:
            return False
        delta = np.abs(obj - basket)
        return bool(delta[0] < 0.12 and delta[1] < 0.12 and -0.05 < obj[2] - basket[2] < 0.20)

    return check


def build_completion_specs(ref: Any, task_id: int, stable_steps: int = 3) -> list[CompletionSpec]:
    """Build all primitive-aligned predicates, including picks and final close."""

    task_id = int(task_id)
    stage = ref.stage_eval
    s = int(stable_steps)
    if s <= 0:
        raise ValueError("stable_steps must be positive")

    if task_id == 1:
        predicates = [
            ("pick cookies", "cookies_grasped_or_lifted>0.035", _pick_predicate(ref, "cookies_1")),
            (
                "place cookies into basket",
                "cookies_in_basket_and_released",
                _released_and(_in_basket(ref, "cookies_1"), "cookies_1"),
            ),
            (
                "pick tomato sauce",
                "tomato_grasped_or_lifted>0.035",
                _pick_predicate(ref, "tomato_sauce_1"),
            ),
            (
                "place tomato into basket",
                "tomato_in_basket_and_released",
                _released_and(_in_basket(ref, "tomato_sauce_1"), "tomato_sauce_1"),
            ),
        ]
    elif task_id == 17:
        predicates: list[tuple[str, str, Predicate]] = [
            (
                "open middle drawer",
                "middle_drawer_displacement>0.10",
                stage._drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10),  # noqa: SLF001
            ),
            ("pick butter", "butter_grasped_or_lifted>0.035", _pick_predicate(ref, "butter_1")),
            (
                "place butter",
                "butter_in_middle_drawer_and_released",
                _released_and(
                    stage._in_drawer_y_window(  # noqa: SLF001
                        "butter_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10
                    ),
                    "butter_1",
                ),
            ),
            (
                "pick chocolate",
                "chocolate_grasped_or_lifted>0.035",
                _pick_predicate(ref, "chocolate_pudding_1"),
            ),
            (
                "place chocolate",
                "chocolate_in_middle_drawer_and_released",
                _released_and(
                    stage._in_drawer_y_window(  # noqa: SLF001
                        "chocolate_pudding_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10
                    ),
                    "chocolate_pudding_1",
                ),
            ),
            (
                "close middle drawer",
                "middle_drawer_displacement<0.08_after_open",
                stage._drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08),  # noqa: SLF001
            ),
        ]
    elif task_id == 20:
        predicates = [
            ("open microwave", "abs(microwave_angle)>0.30", stage._microwave_open(0.30)),  # noqa: SLF001
            ("pick cookies", "cookies_grasped_or_lifted>0.035", _pick_predicate(ref, "cookies_1")),
            (
                "place cookies",
                "cookies_in_microwave_and_released",
                _released_and(stage._in_microwave("cookies_1"), "cookies_1"),  # noqa: SLF001
            ),
            (
                "pick chocolate",
                "chocolate_grasped_or_lifted>0.035",
                _pick_predicate(ref, "chocolate_pudding_1"),
            ),
            (
                "place chocolate",
                "chocolate_in_microwave_and_released",
                _released_and(stage._in_microwave("chocolate_pudding_1"), "chocolate_pudding_1"),  # noqa: SLF001
            ),
            ("close microwave", "abs(microwave_angle)<0.15_after_open", stage._microwave_closed(0.05)),  # noqa: SLF001
        ]
    elif task_id == 22:
        predicates = [
            (
                "pick tomato",
                "tomato_grasped_or_lifted>0.035",
                _pick_predicate(ref, "tomato_sauce_1"),
            ),
            (
                "pour first",
                "first_tilt_cycle_near_cookies",
                stage._body_pour_stage("tomato_sauce_1", "body", "cookies_1"),  # noqa: SLF001
            ),
            (
                "pour second",
                "second_distinct_tilt_cycle_near_cookies",
                stage._body_pour_stage("tomato_sauce_1", "body", "cookies_1"),  # noqa: SLF001
            ),
            (
                "place tomato aside",
                "tomato_near_aside_target_and_released",
                _released_and(
                    stage._near_fixed_position(  # noqa: SLF001
                        "tomato_sauce_1",
                        np.asarray([0.0, -0.2, 0.50], dtype=np.float32),
                        0.20,
                        0.20,
                    ),
                    "tomato_sauce_1",
                ),
            ),
            ("open microwave", "abs(microwave_angle)>0.30", stage._microwave_open(0.30)),  # noqa: SLF001
            ("pick cookies", "cookies_grasped_or_lifted>0.035", _pick_predicate(ref, "cookies_1")),
            (
                "place cookies",
                "cookies_in_microwave_and_released",
                _released_and(stage._in_microwave("cookies_1"), "cookies_1"),  # noqa: SLF001
            ),
            ("close microwave", "abs(microwave_angle)<0.15_after_open", stage._microwave_closed(0.05)),  # noqa: SLF001
        ]
    else:
        raise ValueError(f"Only task 1, 17, 20 and 22 are supported, got task_id={task_id}")

    expected = TASK_PRIMITIVES[task_id]
    labels = tuple(label for label, _, _ in predicates)
    if labels != expected:
        raise RuntimeError(f"Semantic predicate order mismatch for task{task_id}: {labels} != {expected}")
    return [CompletionSpec(label, name, predicate, s) for label, name, predicate in predicates]


class OrderedSemanticTracker:
    """Debounced, latched first-incomplete-primitive state machine."""

    def __init__(self, specs: list[CompletionSpec], *, activation_step: int, state_index: int) -> None:
        if not specs:
            raise ValueError("At least one completion spec is required")
        self.specs = specs
        self.primitives = [PrimitiveState(i, spec.label, spec.predicate_name) for i, spec in enumerate(specs)]
        self.current_index = 0
        self._streak = 0
        self.primitives[0].activated_step = int(activation_step)
        self.primitives[0].activated_state_index = int(state_index)

    @property
    def all_completed(self) -> bool:
        return self.current_index >= len(self.specs)

    @property
    def current_label(self) -> str | None:
        return None if self.all_completed else self.specs[self.current_index].label

    @property
    def completed_count(self) -> int:
        return sum(item.completion_step is not None for item in self.primitives)

    def update(self, env: Any, state: dict[str, Any], step: int) -> dict[str, Any]:
        if self.all_completed:
            return {
                "step": int(step),
                "evaluated_primitive_index": None,
                "evaluated_primitive": None,
                "predicate_value": True,
                "predicate_streak": self._streak,
                "completion_events": [],
                "current_primitive_index": None,
                "current_primitive": None,
                "all_completed": True,
            }

        index = self.current_index
        spec = self.specs[index]
        primitive = self.primitives[index]
        stage_start = int(primitive.activated_state_index or 0)
        value = bool(spec.predicate(env, state, stage_start))
        self._streak = self._streak + 1 if value else 0
        completion_events: list[dict[str, Any]] = []
        evaluated_streak = self._streak

        if self._streak >= spec.stable_steps:
            primitive.completion_step = int(step)
            primitive.completion_state_index = int(state.get("step_idx", step))
            completion_events.append(
                {
                    "event": "primitive_completed",
                    "step": int(step),
                    "state_index": primitive.completion_state_index,
                    "primitive_index": index,
                    "primitive": spec.label,
                    "predicate_name": spec.predicate_name,
                    "stable_steps": spec.stable_steps,
                }
            )
            self.current_index += 1
            self._streak = 0
            if not self.all_completed:
                nxt = self.primitives[self.current_index]
                nxt.activated_step = int(step)
                nxt.activated_state_index = int(state.get("step_idx", step))
                completion_events.append(
                    {
                        "event": "primitive_activated",
                        "step": int(step),
                        "state_index": nxt.activated_state_index,
                        "primitive_index": self.current_index,
                        "primitive": nxt.label,
                    }
                )
            else:
                completion_events.append(
                    {
                        "event": "task_completed",
                        "step": int(step),
                        "state_index": int(state.get("step_idx", step)),
                    }
                )

        return {
            "step": int(step),
            "evaluated_primitive_index": index,
            "evaluated_primitive": spec.label,
            "predicate_name": spec.predicate_name,
            "predicate_value": value,
            "predicate_streak": evaluated_streak,
            "completion_events": completion_events,
            "current_primitive_index": None if self.all_completed else self.current_index,
            "current_primitive": self.current_label,
            "all_completed": self.all_completed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_index": self.current_index,
            "all_completed": self.all_completed,
            "completed_count": self.completed_count,
            "primitives": [asdict(item) for item in self.primitives],
        }


def _round_vector(value: Any, digits: int = 6) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), digits) for item in array]


def physical_snapshot(ref: Any, env: Any, state: dict[str, Any], task_id: int) -> dict[str, Any]:
    """Small auditable state record saved at every action step."""

    objects_by_task = {
        1: ("cookies_1", "tomato_sauce_1"),
        17: ("butter_1", "chocolate_pudding_1"),
        20: ("cookies_1", "chocolate_pudding_1"),
        22: ("tomato_sauce_1", "cookies_1"),
    }
    object_rows: dict[str, Any] = {}
    for object_name in objects_by_task[int(task_id)]:
        object_rows[object_name] = {
            "position": _round_vector(_current_body_pos(ref, env, object_name)),
            "lift_delta": _lift_delta(ref, env, state, object_name),
            "grasped": _grasp_status(env, object_name),
        }

    obs = state.get("last_obs")
    gripper_qpos = obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
    eef_pos = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
    row: dict[str, Any] = {
        "state_index": int(state.get("step_idx", 0)),
        "objects": object_rows,
        "robot0_gripper_qpos": _round_vector(gripper_qpos),
        "robot0_eef_pos": _round_vector(eef_pos),
    }
    if int(task_id) == 1:
        row["basket_position"] = _round_vector(_current_body_pos(ref, env, "basket_1"))
    if int(task_id) == 17:
        current = ref.stage_eval._current_site_pos(env, "wooden_cabinet_1_middle_region")  # noqa: SLF001
        initial = state.get("initial_site_pos", {}).get("wooden_cabinet_1_middle_region")
        row["middle_drawer_region_position"] = _round_vector(current)
        row["middle_drawer_displacement"] = (
            None
            if current is None or initial is None
            else float(np.linalg.norm(np.asarray(current) - np.asarray(initial)))
        )
    if int(task_id) in {20, 22}:
        row["microwave_joint_angle"] = ref.stage_eval._microwave_joint_angle(env)  # noqa: SLF001
        row["microwave_heating_region_position"] = _round_vector(
            ref.stage_eval._current_site_pos(env, "microwave_1_heating_region")  # noqa: SLF001
        )
    if int(task_id) == 22:
        row["tomato_tilt_angle"] = ref.stage_eval._body_tilt_angle(env, "tomato_sauce_1")  # noqa: SLF001
    return row
