#!/usr/bin/env python3
"""Run high-vlm-v2 Tasks 2/3/4 with frozen official physical stages.

The imported high-vlm-v2 planner and rollout loop remain unchanged. This
module only registers physical completion predicates, BDDL resolution, and
auditable state snapshots for tasks the copied runner otherwise rejects.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


EXP_ROOT = Path(os.environ.get("HV2_EXP_ROOT", Path(__file__).resolve().parents[1])).resolve()
PACK_ROOT = EXP_ROOT.parent
SOURCE_VLM_DIR = EXP_ROOT / "source/vlm_ft"
OFFICIAL_BENCHMARK_ROOT = PACK_ROOT / "official_snapshot/evaluation_benchmark"
OFFICIAL_SCORER_PATH = OFFICIAL_BENCHMARK_ROOT / "scripts/task2_26_reference_stage.py"
OFFICIAL_BDDL_DIR = OFFICIAL_BENCHMARK_ROOT / "bddl"
OFFICIAL_SCORER_SHA256 = "0ab5e19cb7b90844b86fe04a76facc0364af55f1e841c4754aa675404a318538"
EXTENSION_TASKS = (2, 3, 4)

TASK_LEGAL_PRIMITIVES: dict[int, tuple[str, ...]] = {
    2: (
        "pick butter",
        "place butter into basket",
        "pick popcorn",
        "place popcorn into basket",
    ),
    3: (
        "pick cream",
        "place cream into basket",
        "pick pudding",
        "place pudding into basket",
    ),
    4: (
        "open top drawer",
        "close the top drawer",
        "open middle drawer",
        "close middle drawer",
        "open bottom drawer",
        "close bottom drawer",
        "open bottom drawer again",
        "open middle drawer again",
        "open top drawer again",
        "place butter into bottom drawer",
        "place butter into middle drawer",
        "place butter into top drawer",
        "close bottom drawer final",
        "close middle drawer final",
        "close top drawer final",
    ),
}

OFFICIAL_OPTIONAL_FINAL_STAGE = {4: "09_Close_Top_Drawer_Final"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required official scorer is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Official scorer SHA-256 mismatch: expected {expected}, got {actual} for {path}"
        )
    return actual


def _verify_bddl_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Official BDDL directory is missing: {path}")
    for task_id in EXTENSION_TASKS:
        matches = sorted(path.glob(f"{task_id}_*.bddl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one official Task{task_id} BDDL under {path}, got {matches}"
            )


def load_official_scorer(path: Path = OFFICIAL_SCORER_PATH) -> ModuleType:
    runtime_dir = path.parent.parent / "openpi_minimal_runtime"
    if not (runtime_dir / "eval_common.py").is_file():
        raise FileNotFoundError(f"Official eval_common.py is missing: {runtime_dir}")
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    spec = importlib.util.spec_from_file_location("task234_frozen_official_stage", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def required_official_stage_specs(scorer: Any, task_id: int) -> list[Any]:
    task_id = int(task_id)
    if task_id not in EXTENSION_TASKS:
        raise ValueError(f"Task{task_id} is not handled by this extension")
    specs = list(scorer._task_specs(task_id))  # noqa: SLF001
    optional_name = OFFICIAL_OPTIONAL_FINAL_STAGE.get(task_id)
    if optional_name is not None:
        scorer_optional = getattr(scorer, "DRAWER_TASK_OPTIONAL_FINAL_STAGE", {}).get(task_id)
        if scorer_optional != optional_name:
            raise RuntimeError(
                f"Official Task{task_id} optional stage mismatch: {scorer_optional!r} != {optional_name!r}"
            )
        specs = [spec for spec in specs if spec.name != optional_name]
    if not specs:
        raise RuntimeError(f"Official Task{task_id} produced no required stages")
    return specs


def _round_vector(value: Any, digits: int = 6) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), digits) for item in array]


def _extension_physical_snapshot(
    ref: Any,
    scorer: Any,
    env: Any,
    state: dict[str, Any],
    task_id: int,
) -> dict[str, Any]:
    objects = {
        2: ("butter_1", "popcorn_1"),
        3: ("cream_cheese_1", "chocolate_pudding_1"),
        4: ("butter_1", "cream_cheese_1"),
    }[int(task_id)]
    obs = state.get("last_obs")
    row: dict[str, Any] = {
        "state_index": int(state.get("step_idx", 0)),
        "robot0_eef_pos": _round_vector(obs.get("robot0_eef_pos") if isinstance(obs, dict) else None),
        "robot0_gripper_qpos": _round_vector(
            obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
        ),
        "objects": {
            name: {"position": _round_vector(ref.ec._body_pos(env, name))}  # noqa: SLF001
            for name in objects
        },
    }
    if int(task_id) in {2, 3}:
        row["basket_position"] = _round_vector(ref.ec._body_pos(env, "basket_1"))  # noqa: SLF001
    if int(task_id) == 4:
        row["drawer_regions"] = {
            drawer: _round_vector(scorer._current_site_pos(env, f"wooden_cabinet_1_{drawer}_region"))  # noqa: SLF001
            for drawer in ("top", "middle", "bottom")
        }
    return row


def install_extension(
    runner: Any,
    *,
    scorer_path: Path = OFFICIAL_SCORER_PATH,
    bddl_dir: Path = OFFICIAL_BDDL_DIR,
    expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
    scorer_module: Any | None = None,
) -> Any:
    scorer_path = scorer_path.resolve()
    bddl_dir = bddl_dir.resolve()
    verify_sha256(scorer_path, expected_scorer_sha256)
    _verify_bddl_dir(bddl_dir)
    scorer = scorer_module if scorer_module is not None else load_official_scorer(scorer_path)

    original_build_completion_specs = runner.build_completion_specs
    original_physical_snapshot = runner.physical_snapshot
    original_build_handoff_windows = runner.build_handoff_windows
    original_load_reference_runtime = runner._load_reference_runtime

    runner.PHYSICAL_METRIC_TASKS = tuple(
        sorted(set(runner.PHYSICAL_METRIC_TASKS).union(EXTENSION_TASKS))
    )
    runner.TASK_PRIMITIVES.update(TASK_LEGAL_PRIMITIVES)

    def build_completion_specs(ref: Any, task_id: int, stable_steps: int = 3) -> list[Any]:
        if int(task_id) not in EXTENSION_TASKS:
            return original_build_completion_specs(ref, task_id, stable_steps=stable_steps)
        from task_semantics import CompletionSpec

        return [
            CompletionSpec(
                label=spec.name,
                predicate_name=f"official_d9f83ac:{spec.name}",
                predicate=spec.check_fn,
                stable_steps=1,
            )
            for spec in required_official_stage_specs(scorer, int(task_id))
        ]

    def physical_snapshot(ref: Any, env: Any, state: dict[str, Any], task_id: int) -> dict[str, Any]:
        if int(task_id) not in EXTENSION_TASKS:
            return original_physical_snapshot(ref, env, state, task_id)
        return _extension_physical_snapshot(ref, scorer, env, state, int(task_id))

    def build_handoff_windows(task_id: int, *args: Any, **kwargs: Any) -> list[Any]:
        if int(task_id) in EXTENSION_TASKS:
            return []
        return original_build_handoff_windows(task_id, *args, **kwargs)

    def load_reference_runtime(path: Path) -> Any:
        ref = original_load_reference_runtime(path)
        ref.ec.DEFAULT_BDDL_BASE = bddl_dir
        return ref

    runner.build_completion_specs = build_completion_specs
    runner.physical_snapshot = physical_snapshot
    runner.build_handoff_windows = build_handoff_windows
    runner._load_reference_runtime = load_reference_runtime
    return scorer


def main() -> None:
    verify_sha256(OFFICIAL_SCORER_PATH, OFFICIAL_SCORER_SHA256)
    _verify_bddl_dir(OFFICIAL_BDDL_DIR)
    if str(SOURCE_VLM_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_VLM_DIR))
    import eval_three_tasks as runner

    install_extension(runner)
    runner.main()


if __name__ == "__main__":
    main()
