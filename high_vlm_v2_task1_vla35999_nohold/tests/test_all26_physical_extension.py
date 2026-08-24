from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from extensions.eval_all26_physical_two_call import (
    ALL_TASKS,
    OFFICIAL_BDDL_DIR,
    OFFICIAL_SCORER_PATH,
    OFFICIAL_SCORER_SHA256,
    install_extension,
    install_physical_extension,
    load_two_call_installer,
    load_official_scorer,
    required_official_stage_specs,
    verify_bddl_dir,
    verify_sha256,
    verify_snapshot,
)


def test_latest_snapshot_and_all_bddls_verify() -> None:
    verified = verify_snapshot()

    assert verified[OFFICIAL_SCORER_PATH] == OFFICIAL_SCORER_SHA256
    mapping = verify_bddl_dir(OFFICIAL_BDDL_DIR)
    assert tuple(mapping) == ALL_TASKS
    assert all(path.name.startswith(f"{task_id}_") for task_id, path in mapping.items())


def test_latest_scorer_has_required_stages_for_all_tasks() -> None:
    scorer = load_official_scorer()

    assert scorer._build_initial_state.__module__.endswith("all26_cc156e5_official_stage")
    assert '"shared_pour_counters"' in inspect.getsource(scorer._build_initial_state)
    for task_id in ALL_TASKS:
        raw = list(scorer._task_specs(task_id))
        required = required_official_stage_specs(scorer, task_id)
        assert raw
        assert required
        optional = scorer.DRAWER_TASK_OPTIONAL_FINAL_STAGE.get(task_id)
        assert optional not in [spec.name for spec in required]


def test_counting_tasks_use_latest_persistent_counter() -> None:
    scorer = load_official_scorer()

    assert scorer.SharedPourCounter.__module__ == "shared_pour_counter"
    assert scorer.COUNTING_POUR_TASKS == {6, 7, 8, 9, 10, 15, 16, 22}
    assert scorer._extra_pour_check(6) is not None


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "scorer.py"
    path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_physical_install_registers_all_tasks_without_prompt_injection(tmp_path: Path) -> None:
    scorer_path = tmp_path / "task2_26_reference_stage.py"
    scorer_path.write_text("fixture\n", encoding="utf-8")
    scorer_sha = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    bddl_dir = tmp_path / "bddl"
    bddl_dir.mkdir()
    for task_id in ALL_TASKS:
        (bddl_dir / f"{task_id}_fixture.bddl").write_text("fixture\n", encoding="utf-8")

    fake_scorer = SimpleNamespace(
        DRAWER_TASK_OPTIONAL_FINAL_STAGE={},
        _task_specs=lambda task_id: [
            SimpleNamespace(name=f"task{task_id}_stage", check_fn=lambda *_: False)
        ],
    )
    fake_ref = SimpleNamespace(
        ec=SimpleNamespace(DEFAULT_BDDL_BASE=Path("/old")),
        stage_eval=SimpleNamespace(name="old"),
    )
    original_handoff = lambda *_args, **_kwargs: ["old"]
    runner = SimpleNamespace(
        PHYSICAL_METRIC_TASKS=(1, 17, 20, 22),
        TASK_PRIMITIVES={},
        build_completion_specs=lambda *_args, **_kwargs: ["old"],
        physical_snapshot=lambda *_args, **_kwargs: {"old": True},
        build_handoff_windows=original_handoff,
        _load_reference_runtime=lambda _path: fake_ref,
    )
    legal = {task_id: (f"task{task_id} primitive",) for task_id in ALL_TASKS}

    install_physical_extension(
        runner,
        scorer_path=scorer_path,
        bddl_dir=bddl_dir,
        expected_scorer_sha256=scorer_sha,
        scorer_module=fake_scorer,
        legal_primitives=legal,
    )

    assert runner.PHYSICAL_METRIC_TASKS == ALL_TASKS
    assert runner.TASK_PRIMITIVES == legal
    assert runner.build_handoff_windows(8) == []
    assert runner._load_reference_runtime(Path("unused")) is fake_ref
    assert fake_ref.ec.DEFAULT_BDDL_BASE == bddl_dir.resolve()
    assert fake_ref.stage_eval is fake_scorer


def test_composed_install_adds_two_call_controller(tmp_path: Path) -> None:
    scorer_path = tmp_path / "task2_26_reference_stage.py"
    scorer_path.write_text("fixture\n", encoding="utf-8")
    scorer_sha = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    bddl_dir = tmp_path / "bddl"
    bddl_dir.mkdir()
    for task_id in ALL_TASKS:
        (bddl_dir / f"{task_id}_fixture.bddl").write_text("fixture\n", encoding="utf-8")
    fake_scorer = SimpleNamespace(
        DRAWER_TASK_OPTIONAL_FINAL_STAGE={},
        _task_specs=lambda task_id: [
            SimpleNamespace(name=f"task{task_id}_stage", check_fn=lambda *_: False)
        ],
    )
    fake_ref = SimpleNamespace(ec=SimpleNamespace(DEFAULT_BDDL_BASE=Path("/old")))
    runner = SimpleNamespace(
        PHYSICAL_METRIC_TASKS=(),
        TASK_PRIMITIVES={},
        build_completion_specs=lambda *_args, **_kwargs: [],
        physical_snapshot=lambda *_args, **_kwargs: {},
        build_handoff_windows=lambda *_args, **_kwargs: [],
        _load_reference_runtime=lambda _path: fake_ref,
    )
    calls: list[int] = []

    install_extension(
        runner,
        required_calls=2,
        scorer_path=scorer_path,
        bddl_dir=bddl_dir,
        expected_scorer_sha256=scorer_sha,
        scorer_module=fake_scorer,
        legal_primitives={task_id: (str(task_id),) for task_id in ALL_TASKS},
        two_call_installer=lambda _runner, required_calls: calls.append(required_calls),
    )

    assert calls == [2]


def test_two_call_installer_can_load_from_frozen_path(tmp_path: Path) -> None:
    module_path = tmp_path / "frozen_two_call.py"
    module_path.write_text(
        "def install_extension(runner, required_calls):\n"
        "    runner.loaded_required_calls = required_calls\n",
        encoding="utf-8",
    )
    runner = SimpleNamespace()

    installer = load_two_call_installer(module_path)
    installer(runner, required_calls=2)

    assert runner.loaded_required_calls == 2
