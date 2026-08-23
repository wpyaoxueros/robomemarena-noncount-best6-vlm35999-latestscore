from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from extensions.eval_task234_physical import (
    OFFICIAL_OPTIONAL_FINAL_STAGE,
    TASK_LEGAL_PRIMITIVES,
    install_extension,
    required_official_stage_specs,
    verify_sha256,
)


@dataclass(frozen=True)
class FakeStageSpec:
    name: str
    check_fn: object


class FakeOfficialScorer:
    DRAWER_TASK_OPTIONAL_FINAL_STAGE = {4: "09_Close_Top_Drawer_Final"}

    @staticmethod
    def _task_specs(task_id: int) -> list[FakeStageSpec]:
        names = {
            2: ["01_Place_Butter_Basket", "02_Place_Popcorn_Basket"],
            3: ["01_Place_Cream_Basket", "02_Place_Pudding_Basket"],
            4: [
                "01_Open_Top_Drawer",
                "02_Close_Top_Drawer",
                "03_Open_Middle_Drawer",
                "04_Close_Middle_Drawer",
                "05_Open_Bottom_Drawer",
                "06_Close_Bottom_Drawer",
                "07_Open_Top_Drawer_Again",
                "08_Put_Butter_Top_Drawer",
                "09_Close_Top_Drawer_Final",
            ],
        }[int(task_id)]
        return [FakeStageSpec(name, lambda *_: False) for name in names]


def test_task2_and_task3_use_official_placement_stages() -> None:
    scorer = FakeOfficialScorer()
    assert [row.name for row in required_official_stage_specs(scorer, 2)] == [
        "01_Place_Butter_Basket",
        "02_Place_Popcorn_Basket",
    ]
    assert [row.name for row in required_official_stage_specs(scorer, 3)] == [
        "01_Place_Cream_Basket",
        "02_Place_Pudding_Basket",
    ]


def test_task4_final_close_is_optional() -> None:
    scorer = FakeOfficialScorer()
    required = [row.name for row in required_official_stage_specs(scorer, 4)]
    assert required == [
        "01_Open_Top_Drawer",
        "02_Close_Top_Drawer",
        "03_Open_Middle_Drawer",
        "04_Close_Middle_Drawer",
        "05_Open_Bottom_Drawer",
        "06_Close_Bottom_Drawer",
        "07_Open_Top_Drawer_Again",
        "08_Put_Butter_Top_Drawer",
    ]
    assert OFFICIAL_OPTIONAL_FINAL_STAGE[4] == "09_Close_Top_Drawer_Final"


def test_legal_primitives_match_training_prompt_contract() -> None:
    assert TASK_LEGAL_PRIMITIVES[2] == (
        "pick butter",
        "place butter into basket",
        "pick popcorn",
        "place popcorn into basket",
    )
    assert TASK_LEGAL_PRIMITIVES[3] == (
        "pick cream",
        "place cream into basket",
        "pick pudding",
        "place pudding into basket",
    )
    assert "open top drawer again" in TASK_LEGAL_PRIMITIVES[4]
    assert "place butter into top drawer" in TASK_LEGAL_PRIMITIVES[4]


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    scorer = tmp_path / "task2_26_reference_stage.py"
    scorer.write_text("official scorer fixture\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_sha256(scorer, "0" * 64)


def test_install_extension_changes_only_registration_boundaries(tmp_path: Path) -> None:
    scorer_path = tmp_path / "task2_26_reference_stage.py"
    scorer_path.write_text("official scorer fixture\n", encoding="utf-8")
    expected_sha = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    bddl_dir = tmp_path / "bddl"
    bddl_dir.mkdir()
    for task_id in (2, 3, 4):
        (bddl_dir / f"{task_id}_fixture.bddl").write_text("fixture\n", encoding="utf-8")

    original_completion = lambda *_args, **_kwargs: ["original"]
    original_snapshot = lambda *_args, **_kwargs: {"original": True}
    original_handoff = lambda *_args, **_kwargs: ["original"]
    fake_ref = SimpleNamespace(ec=SimpleNamespace(DEFAULT_BDDL_BASE=Path("/original")))
    runner = SimpleNamespace(
        PHYSICAL_METRIC_TASKS=(1, 17, 20, 22),
        TASK_PRIMITIVES={},
        build_completion_specs=original_completion,
        physical_snapshot=original_snapshot,
        build_handoff_windows=original_handoff,
        _load_reference_runtime=lambda _path: fake_ref,
    )

    install_extension(
        runner,
        scorer_path=scorer_path,
        bddl_dir=bddl_dir,
        expected_scorer_sha256=expected_sha,
        scorer_module=FakeOfficialScorer(),
    )

    assert runner.PHYSICAL_METRIC_TASKS == (1, 2, 3, 4, 17, 20, 22)
    assert runner.TASK_PRIMITIVES[2] == TASK_LEGAL_PRIMITIVES[2]
    assert runner.build_completion_specs is not original_completion
    assert runner.physical_snapshot is not original_snapshot
    assert runner.build_handoff_windows is not original_handoff
    assert runner._load_reference_runtime(Path("unused")) is fake_ref
    assert fake_ref.ec.DEFAULT_BDDL_BASE == bddl_dir.resolve()
