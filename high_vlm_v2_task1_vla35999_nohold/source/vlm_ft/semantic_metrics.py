"""Metrics and label normalization for online VLM task-segmentation evaluation.

The module is deliberately independent from LIBERO and model code.  It can be
unit-tested with a sequence of timestamped predictions and semantic boundary
annotations produced by any rollout implementation.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


UNKNOWN_PRIMITIVE = "__unknown__"


TASK_PRIMITIVES: dict[int, tuple[str, ...]] = {
    1: (
        "pick cookies",
        "place cookies into basket",
        "pick tomato sauce",
        "place tomato into basket",
    ),
    17: (
        "open middle drawer",
        "pick butter",
        "place butter",
        "pick chocolate",
        "place chocolate",
        "close middle drawer",
    ),
    20: (
        "open microwave",
        "pick cookies",
        "place cookies",
        "pick chocolate",
        "place chocolate",
        "close microwave",
    ),
    22: (
        "pick tomato",
        "pour first",
        "pour second",
        "place tomato aside",
        "open microwave",
        "pick cookies",
        "place cookies",
        "close microwave",
    ),
}


_EXTRA_ALIASES: dict[int, dict[str, tuple[str, ...]]] = {
    1: {
        "pick cookies": ("pick up cookies", "pick up the cookies", "grasp cookies"),
        "place cookies into basket": (
            "place the cookies into the basket",
            "put cookies in basket",
            "put the cookies in the basket",
        ),
        "pick tomato sauce": (
            "pick tomato",
            "pick up tomato sauce",
            "pick up the tomato sauce",
            "grasp tomato sauce",
        ),
        "place tomato into basket": (
            "place tomato sauce into basket",
            "place the tomato sauce into the basket",
            "put tomato sauce in basket",
            "put the tomato sauce in the basket",
        ),
    },
    17: {
        "open middle drawer": ("open the middle drawer",),
        "pick butter": ("pick up butter", "pick up the butter", "grasp butter", "grasp the butter"),
        "place butter": (
            "place the butter",
            "put butter in middle drawer",
            "put the butter in the middle drawer",
            "place butter in middle drawer",
            "place the butter in the middle drawer",
        ),
        "pick chocolate": (
            "pick up chocolate",
            "pick up the chocolate",
            "pick chocolate pudding",
            "pick up chocolate pudding",
        ),
        "place chocolate": (
            "place the chocolate",
            "place chocolate pudding",
            "put chocolate in middle drawer",
            "put the chocolate in the middle drawer",
            "place chocolate in middle drawer",
            "place the chocolate in the middle drawer",
        ),
        "close middle drawer": ("close the middle drawer",),
    },
    20: {
        "open microwave": ("open the microwave", "open microwave door", "open the microwave door"),
        "pick cookies": ("pick cookies up", "pick up cookies", "pick up the cookies", "grasp cookies"),
        "place cookies": (
            "place the cookies",
            "put cookies in microwave",
            "put the cookies in the microwave",
            "place cookies in microwave",
            "place the cookies in the microwave",
        ),
        "pick chocolate": (
            "pick up chocolate",
            "pick up the chocolate",
            "pick chocolate pudding",
            "pick up chocolate pudding",
        ),
        "place chocolate": (
            "place the chocolate",
            "place chocolate pudding",
            "put chocolate in microwave",
            "put the chocolate in the microwave",
            "place chocolate in microwave",
            "place the chocolate in the microwave",
        ),
        "close microwave": ("close the microwave", "close microwave door", "close the microwave door"),
    },
    22: {
        "pick tomato": (
            "pick tomato sauce",
            "pick up tomato",
            "pick up the tomato",
            "pick up tomato sauce",
            "pick up the tomato sauce",
        ),
        "pour first": (
            "first pour",
            "pour once",
            "pour tomato sauce first",
            "pour tomato sauce over cookies first time",
            "pour sauce over cookies first time",
        ),
        "pour second": (
            "second pour",
            "pour twice",
            "pour tomato sauce second",
            "pour tomato sauce over cookies second time",
            "pour sauce over cookies second time",
        ),
        "place tomato aside": (
            "place tomato sauce aside",
            "put tomato aside",
            "put tomato sauce aside",
            "set tomato aside",
            "set tomato sauce aside",
        ),
        "open microwave": ("open the microwave", "open microwave door", "open the microwave door"),
        "pick cookies": ("pick up cookies", "pick up the cookies", "grasp cookies"),
        "place cookies": (
            "place the cookies",
            "put cookies in microwave",
            "put the cookies in the microwave",
            "place cookies in microwave",
            "place the cookies in the microwave",
        ),
        "close microwave": ("close the microwave", "close microwave door", "close the microwave door"),
    },
}


def _surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


class PrimitiveNormalizer:
    """Conservative exact-alias mapper from free text to a canonical primitive."""

    def __init__(self, task_id: int, primitives: Sequence[str] | None = None) -> None:
        self.task_id = int(task_id)
        self.primitives = tuple(primitives or TASK_PRIMITIVES[self.task_id])
        aliases: dict[str, str] = {}
        extras = _EXTRA_ALIASES.get(self.task_id, {})
        for primitive in self.primitives:
            candidates = (primitive, *extras.get(primitive, ()))
            for candidate in candidates:
                key = _surface(candidate)
                previous = aliases.get(key)
                if previous is not None and previous != primitive:
                    raise ValueError(f"Ambiguous primitive alias {candidate!r}: {previous!r} vs {primitive!r}")
                aliases[key] = primitive
        self._aliases = aliases

    def normalize(self, text: str) -> tuple[str, str]:
        key = _surface(text)
        if key in self._aliases:
            canonical = self._aliases[key]
            method = "canonical" if key == _surface(canonical) else "explicit_alias"
            return canonical, method
        return UNKNOWN_PRIMITIVE, "unmapped"


@dataclass(frozen=True)
class PredictionRecord:
    call_index: int
    input_step: int
    parsed_primitive: str
    normalized_primitive: str
    applied_step: int | None = None


@dataclass(frozen=True)
class BoundaryRecord:
    task_id: int
    episode: int
    seed: int
    boundary_index: int
    old_primitive: str
    next_primitive: str
    activated_step: int | None
    completion_step: int | None
    next_completion_step: int | None
    episode_end_step: int
    boundary_source: str = "original_trajectory"


def _first_stable_start(
    predictions: Sequence[PredictionRecord],
    target: str,
    stable_calls: int,
    *,
    allowed_start_offsets: set[int] | None = None,
) -> int | None:
    if stable_calls <= 0:
        raise ValueError("stable_calls must be positive")
    for start in range(0, len(predictions) - stable_calls + 1):
        if allowed_start_offsets is not None and start not in allowed_start_offsets:
            continue
        window = predictions[start : start + stable_calls]
        if all(item.normalized_primitive == target for item in window):
            return start
    return None


def evaluate_boundary(
    boundary: BoundaryRecord,
    predictions: Sequence[PredictionRecord],
    *,
    stable_calls: int = 2,
    scna_tolerance_calls: int = 2,
    premature_tolerance_steps: int = 1,
) -> dict[str, Any]:
    """Evaluate one ordered old->next boundary.

    SCNA@2 follows the convention discussed for this experiment: the stable
    two-call run may start at k0, k1, or k2.  Consequently a definitive
    negative needs observations through k3.
    """

    ordered = sorted(predictions, key=lambda item: item.call_index)
    activated = boundary.activated_step
    completion = boundary.completion_step
    end = boundary.next_completion_step
    if end is None:
        end = boundary.episode_end_step + 1

    active_predictions = [
        item
        for item in ordered
        if activated is not None and item.input_step >= activated and item.input_step < end
    ]

    if completion is None:
        post: list[PredictionRecord] = []
    else:
        post = [item for item in active_predictions if item.input_step >= completion]

    scna0: bool | None
    if completion is None or not post:
        scna0 = None
    else:
        scna0 = post[0].normalized_primitive == boundary.next_primitive

    allowed_starts = set(range(scna_tolerance_calls + 1))
    scna2_start = _first_stable_start(
        post,
        boundary.next_primitive,
        stable_calls,
        allowed_start_offsets=allowed_starts,
    )
    required_for_definitive_negative = scna_tolerance_calls + stable_calls
    if completion is None:
        scna2: bool | None = None
    elif scna2_start is not None:
        scna2 = True
    elif len(post) >= required_for_definitive_negative:
        scna2 = False
    else:
        scna2 = None

    stable_start = _first_stable_start(post, boundary.next_primitive, stable_calls)
    stable_switch = stable_start is not None
    stable_switch_input_step = post[stable_start].input_step if stable_start is not None else None
    stable_switch_applied_step = post[stable_start].applied_step if stable_start is not None else None
    latency_steps = (
        stable_switch_input_step - completion
        if stable_switch_input_step is not None and completion is not None
        else None
    )
    latency_calls = stable_start if stable_start is not None else None
    applied_latency_steps = (
        stable_switch_applied_step - completion
        if stable_switch_applied_step is not None and completion is not None
        else None
    )

    premature_end = boundary.episode_end_step + 1 if completion is None else completion - premature_tolerance_steps
    premature_predictions = [
        item
        for item in active_predictions
        if activated is not None and item.input_step >= activated and item.input_step < premature_end
    ]
    premature_start = _first_stable_start(premature_predictions, boundary.next_primitive, stable_calls)
    premature = premature_start is not None

    regression_start: int | None = None
    regression_count = 0
    if stable_start is not None:
        after_stable = post[stable_start + stable_calls :]
        old_streak = 0
        for start, item in enumerate(after_stable):
            if item.normalized_primitive == boundary.old_primitive:
                old_streak += 1
            else:
                old_streak = 0
            if old_streak == stable_calls:
                regression_count += 1
                if regression_start is None:
                    regression_start = start - stable_calls + 1

    return {
        **asdict(boundary),
        "num_active_vlm_calls": len(active_predictions),
        "num_post_completion_vlm_calls": len(post),
        "first_post_completion_call_index": post[0].call_index if post else None,
        "first_post_completion_input_step": post[0].input_step if post else None,
        "first_post_completion_prediction": post[0].normalized_primitive if post else None,
        "scna0": scna0,
        "scna2": scna2,
        "scna2_stable_start_offset": scna2_start,
        "stable_switch": stable_switch,
        "stable_switch_call_index": post[stable_start].call_index if stable_start is not None else None,
        "stable_switch_input_step": stable_switch_input_step,
        "stable_switch_applied_step": stable_switch_applied_step,
        "switch_latency_steps": latency_steps,
        "switch_latency_calls": latency_calls,
        "switch_latency_applied_steps": applied_latency_steps,
        "premature_switch": premature,
        "premature_stable_start_call_index": (
            premature_predictions[premature_start].call_index if premature_start is not None else None
        ),
        "post_switch_regression": regression_start is not None,
        "regression_count": regression_count,
        "stable_calls_required": stable_calls,
        "scna_tolerance_calls": scna_tolerance_calls,
        "premature_tolerance_steps": premature_tolerance_steps,
        "post_completion_predictions": [
            {
                "call_index": item.call_index,
                "input_step": item.input_step,
                "applied_step": item.applied_step,
                "prediction": item.normalized_primitive,
            }
            for item in post
        ],
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scna0_observed = [row for row in records if row.get("scna0") is not None]
    scna2_observed = [row for row in records if row.get("scna2") is not None]
    activated = [row for row in records if row.get("activated_step") is not None]
    completed = [row for row in records if row.get("completion_step") is not None]
    stable = [row for row in records if row.get("stable_switch")]
    latency_steps = [float(row["switch_latency_steps"]) for row in stable if row.get("switch_latency_steps") is not None]
    latency_calls = [float(row["switch_latency_calls"]) for row in stable if row.get("switch_latency_calls") is not None]
    applied_latency_steps = [
        float(row["switch_latency_applied_steps"])
        for row in stable
        if row.get("switch_latency_applied_steps") is not None
    ]
    regression_denominator = stable

    return {
        "num_boundaries": len(records),
        "num_activated_boundaries": len(activated),
        "num_completed_boundaries": len(completed),
        "scna0": _safe_rate(sum(bool(row["scna0"]) for row in scna0_observed), len(scna0_observed)),
        "scna0_numerator": sum(bool(row["scna0"]) for row in scna0_observed),
        "scna0_denominator": len(scna0_observed),
        "scna2": _safe_rate(sum(bool(row["scna2"]) for row in scna2_observed), len(scna2_observed)),
        "scna2_numerator": sum(bool(row["scna2"]) for row in scna2_observed),
        "scna2_denominator": len(scna2_observed),
        "stable_switch_coverage": _safe_rate(len(stable), len(completed)),
        "stable_switch_numerator": len(stable),
        "stable_switch_denominator": len(completed),
        "premature_switch_rate": _safe_rate(
            sum(bool(row["premature_switch"]) for row in activated), len(activated)
        ),
        "premature_switch_numerator": sum(bool(row["premature_switch"]) for row in activated),
        "premature_switch_denominator": len(activated),
        "post_switch_regression_rate": _safe_rate(
            sum(bool(row["post_switch_regression"]) for row in regression_denominator),
            len(regression_denominator),
        ),
        "post_switch_regression_numerator": sum(
            bool(row["post_switch_regression"]) for row in regression_denominator
        ),
        "post_switch_regression_denominator": len(regression_denominator),
        "switch_latency_steps_mean": statistics.fmean(latency_steps) if latency_steps else None,
        "switch_latency_steps_median": statistics.median(latency_steps) if latency_steps else None,
        "switch_latency_steps_p90": _percentile(latency_steps, 0.90),
        "switch_latency_calls_mean": statistics.fmean(latency_calls) if latency_calls else None,
        "switch_latency_calls_median": statistics.median(latency_calls) if latency_calls else None,
        "switch_latency_calls_p90": _percentile(latency_calls, 0.90),
        "switch_latency_applied_steps_mean": (
            statistics.fmean(applied_latency_steps) if applied_latency_steps else None
        ),
        "switch_latency_applied_steps_median": (
            statistics.median(applied_latency_steps) if applied_latency_steps else None
        ),
        "switch_latency_applied_steps_p90": _percentile(applied_latency_steps, 0.90),
    }


def _mean_non_null(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.fmean(kept) if kept else None


def aggregate_boundary_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_task_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_boundary_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_id = int(row["task_id"])
        per_task_rows[task_id].append(row)
        key = f"task{task_id}:{row['old_primitive']}->{row['next_primitive']}"
        per_boundary_rows[key].append(row)

    per_task = {str(task_id): summarize_records(rows) for task_id, rows in sorted(per_task_rows.items())}
    per_boundary = {key: summarize_records(rows) for key, rows in sorted(per_boundary_rows.items())}
    macro_fields = (
        "scna0",
        "scna2",
        "stable_switch_coverage",
        "premature_switch_rate",
        "post_switch_regression_rate",
        "switch_latency_steps_mean",
        "switch_latency_steps_median",
        "switch_latency_calls_mean",
        "switch_latency_calls_median",
        "switch_latency_applied_steps_mean",
        "switch_latency_applied_steps_median",
    )
    macro = {field: _mean_non_null(summary.get(field) for summary in per_task.values()) for field in macro_fields}
    return {
        "micro": summarize_records(records),
        "macro_over_tasks": macro,
        "per_task": per_task,
        "per_boundary": per_boundary,
    }
