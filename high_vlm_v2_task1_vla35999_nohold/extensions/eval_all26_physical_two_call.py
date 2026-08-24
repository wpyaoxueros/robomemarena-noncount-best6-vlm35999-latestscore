#!/usr/bin/env python3
"""Run Task1-26 with cc156e5 physical stages and two-call prompt commit."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np


EXP_ROOT = Path(os.environ.get("HV2_EXP_ROOT", Path(__file__).resolve().parents[1])).resolve()
SOURCE_VLM_DIR = EXP_ROOT / "source/vlm_ft"
SNAPSHOT_ROOT = EXP_ROOT / "official_snapshot_cc156e5"
OFFICIAL_BENCHMARK_ROOT = SNAPSHOT_ROOT / "evaluation_benchmark"
OFFICIAL_SCORER_PATH = OFFICIAL_BENCHMARK_ROOT / "scripts/task2_26_reference_stage.py"
OFFICIAL_BDDL_DIR = OFFICIAL_BENCHMARK_ROOT / "bddl"
OFFICIAL_SHA256SUMS = SNAPSHOT_ROOT / "SHA256SUMS"
OFFICIAL_SCORER_SHA256 = "4eb949049b3175df01e8c632a6159a4b65bf9e2a667f6cbe612132ba5e7e0b99"
ALL_TASKS = tuple(range(1, 27))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required official file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Official file SHA-256 mismatch: expected {expected}, got {actual} for {path}"
        )
    return actual


def _read_sha256_manifest(path: Path) -> dict[Path, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official SHA-256 manifest is missing: {path}")
    rows: dict[Path, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            digest, relative = line.split(None, 1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed SHA-256 manifest line {line_number}: {raw!r}") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Invalid SHA-256 at manifest line {line_number}: {digest!r}")
        resolved = (path.parent / relative.strip()).resolve()
        if path.parent.resolve() not in resolved.parents:
            raise RuntimeError(f"Manifest path escapes snapshot root: {relative!r}")
        if resolved in rows:
            raise RuntimeError(f"Duplicate manifest path: {relative!r}")
        rows[resolved] = digest
    if not rows:
        raise RuntimeError(f"Official SHA-256 manifest is empty: {path}")
    return rows


def verify_snapshot(path: Path = OFFICIAL_SHA256SUMS) -> dict[Path, str]:
    rows = _read_sha256_manifest(path.resolve())
    for file_path, expected in rows.items():
        verify_sha256(file_path, expected)
    if rows.get(OFFICIAL_SCORER_PATH.resolve()) != OFFICIAL_SCORER_SHA256:
        raise RuntimeError("Official scorer is absent from SHA256SUMS or has an unexpected digest")
    verify_bddl_dir(OFFICIAL_BDDL_DIR)
    return rows


def verify_bddl_dir(path: Path) -> dict[int, Path]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Official BDDL directory is missing: {path}")
    mapping: dict[int, Path] = {}
    for candidate in sorted(path.glob("[0-9]*.bddl")):
        match = re.fullmatch(r"([0-9]+)_.+\.bddl", candidate.name)
        if match is None:
            continue
        task_id = int(match.group(1))
        if task_id in mapping:
            raise RuntimeError(f"Multiple BDDL files found for Task{task_id}: {mapping[task_id]}, {candidate}")
        mapping[task_id] = candidate.resolve()
    if tuple(sorted(mapping)) != ALL_TASKS:
        raise RuntimeError(f"Expected exactly one BDDL for Task1-26, got {sorted(mapping)}")
    return dict(sorted(mapping.items()))


def _load_canonical_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_scorer(path: Path = OFFICIAL_SCORER_PATH) -> ModuleType:
    path = path.resolve()
    scripts_dir = path.parent
    runtime_dir = path.parent.parent / "openpi_minimal_runtime"
    dependencies = {
        "robocerebra_adapter": runtime_dir / "robocerebra_adapter.py",
        "task_prompts": runtime_dir / "task_prompts.py",
        "eval_common": runtime_dir / "eval_common.py",
        "shared_pour_counter": scripts_dir / "shared_pour_counter.py",
    }
    for dependency in dependencies.values():
        if not dependency.is_file():
            raise FileNotFoundError(f"Official scorer dependency is missing: {dependency}")
    for directory in (scripts_dir, runtime_dir):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)

    _load_canonical_module("robocerebra_adapter", dependencies["robocerebra_adapter"])
    _load_canonical_module("task_prompts", dependencies["task_prompts"])
    _load_canonical_module("eval_common", dependencies["eval_common"])
    _load_canonical_module("shared_pour_counter", dependencies["shared_pour_counter"])
    return _load_canonical_module("all26_cc156e5_official_stage", path)


def required_official_stage_specs(scorer: Any, task_id: int) -> list[Any]:
    task_id = int(task_id)
    if task_id not in ALL_TASKS:
        raise ValueError(f"Task{task_id} is outside Task1-26")
    specs = list(scorer._task_specs(task_id))  # noqa: SLF001
    optional = getattr(scorer, "DRAWER_TASK_OPTIONAL_FINAL_STAGE", {}).get(task_id)
    if optional is not None:
        if optional not in [spec.name for spec in specs]:
            raise RuntimeError(f"Task{task_id} optional stage {optional!r} is absent from official specs")
        specs = [spec for spec in specs if spec.name != optional]
    if not specs:
        raise RuntimeError(f"Official Task{task_id} produced no required stages")
    return specs


def _default_legal_primitives() -> dict[int, tuple[str, ...]]:
    if str(SOURCE_VLM_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_VLM_DIR))
    from prompt_contract_26 import TASK_SPECS

    mapping = {
        task_id: tuple(str(value) for value in TASK_SPECS[task_id]["primitives"])
        for task_id in ALL_TASKS
    }
    if tuple(mapping) != ALL_TASKS or any(not values for values in mapping.values()):
        raise RuntimeError("Training prompt contract does not define legal primitives for Task1-26")
    return mapping


def _round_vector(value: Any, digits: int = 6) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), digits) for item in array]


def _physical_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    obs = state.get("last_obs")
    counters: dict[str, Any] = {}
    for key, counter in state.get("shared_pour_counters", {}).items():
        label = "|".join(str(value) for value in key) if isinstance(key, tuple) else str(key)
        counters[label] = {
            "event_count": int(getattr(counter, "event_count", 0)),
            "available": bool(getattr(counter, "available", False)),
            "events": list(getattr(counter, "events", [])),
        }
    return {
        "state_index": int(state.get("step_idx", 0)),
        "robot0_eef_pos": _round_vector(obs.get("robot0_eef_pos") if isinstance(obs, dict) else None),
        "robot0_gripper_qpos": _round_vector(
            obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
        ),
        "shared_pour_counters": counters,
    }


def install_physical_extension(
    runner: Any,
    *,
    scorer_path: Path = OFFICIAL_SCORER_PATH,
    bddl_dir: Path = OFFICIAL_BDDL_DIR,
    expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
    scorer_module: Any | None = None,
    legal_primitives: Mapping[int, tuple[str, ...]] | None = None,
) -> Any:
    scorer_path = scorer_path.resolve()
    bddl_dir = bddl_dir.resolve()
    verify_sha256(scorer_path, expected_scorer_sha256)
    verify_bddl_dir(bddl_dir)
    scorer = scorer_module if scorer_module is not None else load_official_scorer(scorer_path)
    legal = dict(legal_primitives or _default_legal_primitives())
    if tuple(sorted(legal)) != ALL_TASKS:
        raise RuntimeError(f"Legal primitive mapping must cover Task1-26, got {sorted(legal)}")

    original_load_reference_runtime = runner._load_reference_runtime
    runner.PHYSICAL_METRIC_TASKS = ALL_TASKS
    runner.TASK_PRIMITIVES.update(legal)

    def build_completion_specs(_ref: Any, task_id: int, stable_steps: int = 3) -> list[Any]:
        del stable_steps
        from task_semantics import CompletionSpec

        return [
            CompletionSpec(
                label=spec.name,
                predicate_name=f"official_cc156e5:{spec.name}",
                predicate=spec.check_fn,
                stable_steps=1,
            )
            for spec in required_official_stage_specs(scorer, int(task_id))
        ]

    def physical_snapshot(_ref: Any, _env: Any, state: dict[str, Any], _task_id: int) -> dict[str, Any]:
        return _physical_snapshot(state)

    def build_handoff_windows(_task_id: int, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def load_reference_runtime(path: Path) -> Any:
        ref = original_load_reference_runtime(path)
        ref.ec.DEFAULT_BDDL_BASE = bddl_dir
        ref.stage_eval = scorer
        return ref

    runner.build_completion_specs = build_completion_specs
    runner.physical_snapshot = physical_snapshot
    runner.build_handoff_windows = build_handoff_windows
    runner._load_reference_runtime = load_reference_runtime
    return scorer


def install_extension(
    runner: Any,
    *,
    required_calls: int = 2,
    scorer_path: Path = OFFICIAL_SCORER_PATH,
    bddl_dir: Path = OFFICIAL_BDDL_DIR,
    expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
    scorer_module: Any | None = None,
    legal_primitives: Mapping[int, tuple[str, ...]] | None = None,
    two_call_installer: Callable[..., Any] | None = None,
) -> Any:
    scorer = install_physical_extension(
        runner,
        scorer_path=scorer_path,
        bddl_dir=bddl_dir,
        expected_scorer_sha256=expected_scorer_sha256,
        scorer_module=scorer_module,
        legal_primitives=legal_primitives,
    )
    if two_call_installer is None:
        if str(EXP_ROOT) not in sys.path:
            sys.path.insert(0, str(EXP_ROOT))
        from extensions.eval_two_call_prompt_commit import install_extension as two_call_installer
    two_call_installer(runner, required_calls=int(required_calls))
    return scorer


def _required_calls_from_env() -> int:
    raw = os.environ.get("CONTROLLER_STABLE_VLM_CALLS", "2")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"CONTROLLER_STABLE_VLM_CALLS must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError("CONTROLLER_STABLE_VLM_CALLS must be positive")
    return value


def main() -> None:
    verify_snapshot()
    if str(SOURCE_VLM_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_VLM_DIR))
    import eval_three_tasks as runner

    install_extension(runner, required_calls=_required_calls_from_env())
    runner.main()


if __name__ == "__main__":
    main()
