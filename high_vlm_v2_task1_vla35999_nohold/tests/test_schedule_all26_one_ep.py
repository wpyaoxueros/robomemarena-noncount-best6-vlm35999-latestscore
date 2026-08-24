from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.schedule_all26_one_ep import (
    EventLedger,
    AttemptResult,
    SharedExclusions,
    classify_attempt,
    ensure_contract,
    load_assignment,
    merge_excluded_nodes,
    parse_tasks,
    read_summary_rows,
)


LAUNCHER = Path(__file__).parents[1] / "scripts" / "launch_all26_one_ep_account.sh"


def test_parse_tasks_is_sorted_unique_and_bounded() -> None:
    assert parse_tasks("3,1-2,2,26") == (1, 2, 3, 26)
    with pytest.raises(ValueError):
        parse_tasks("0,1")
    with pytest.raises(ValueError):
        parse_tasks("4-2")


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True)
    fieldnames = ["task_id", "episode", "seed", "task_success", "failure_reason"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_policy_failure_with_one_summary_row_is_valid(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_summary(
        run_root / "eval" / "summary.tsv",
        [
            {
                "task_id": "7",
                "episode": "0",
                "seed": "104",
                "task_success": "0",
                "failure_reason": "max_steps",
            }
        ],
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps({"hostname": "ACD1-13", "task": 7, "seed": 104, "run_id": "run"}),
        encoding="utf-8",
    )

    result = classify_attempt(
        task=7,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=run_root,
        returncode=0,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )

    assert result.valid is True
    assert result.task_success == "0"
    assert result.node == "ACD1-13"
    assert result.summary_sha256


def test_header_only_summary_is_invalid_runtime_attempt(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_summary(run_root / "eval" / "summary.tsv", [])
    (run_root / "run_manifest.json").write_text(
        json.dumps({"hostname": "ACD1-54", "task": 1, "seed": 104, "run_id": "run"}),
        encoding="utf-8",
    )

    result = classify_attempt(
        task=1,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=run_root,
        returncode=134,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )

    assert result.valid is False
    assert result.node == "ACD1-54"
    assert result.failure_reason == "launcher_exit:134:no_summary_row"


def test_truncated_summary_row_fails_closed(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_summary(
        run_root / "eval" / "summary.tsv",
        [{"task_id": "1", "episode": "0", "seed": "104", "task_success": "", "failure_reason": ""}],
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps({"hostname": "ACD1-13", "task": 1, "seed": 104, "run_id": "run"}),
        encoding="utf-8",
    )
    result = classify_attempt(
        task=1,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=run_root,
        returncode=0,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )
    assert result.valid is False
    assert "summary.task_success" in result.failure_reason


def test_allocation_identity_supplies_node_before_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "allocation_identity.json").write_text(
        json.dumps({"hostname": "ACD1-29", "task": 1, "seed": 104, "run_id": "run"}),
        encoding="utf-8",
    )
    result = classify_attempt(
        task=1,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=run_root,
        returncode=1,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )
    assert result.valid is False
    assert result.node == "ACD1-29"


def test_more_than_one_summary_row_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "eval" / "summary.tsv"
    write_summary(
        path,
        [
            {"task_id": "1", "episode": "0", "seed": "104", "task_success": "0", "failure_reason": "x"},
            {"task_id": "1", "episode": "1", "seed": "104", "task_success": "1", "failure_reason": ""},
        ],
    )
    assert len(read_summary_rows(path)) == 2

    result = classify_attempt(
        task=1,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=tmp_path,
        returncode=0,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )
    assert result.valid is False
    assert result.failure_reason == "invalid_summary_row_count:2"


def test_event_ledger_resume_counts_only_valid_attempts(tmp_path: Path) -> None:
    ledger = EventLedger(tmp_path / "events.jsonl")
    ledger.append({"event": "attempt_finished", "task": 1, "valid": False})
    valid = AttemptResult(
        task=2,
        attempt=1,
        run_id="run2",
        valid=True,
        returncode=0,
        node="ACD1-13",
        summary_path="/summary.tsv",
        summary_sha256="a" * 64,
        manifest_path="/manifest.json",
        manifest_sha256="b" * 64,
        launcher_log_path="/launcher.log",
        launcher_log_sha256="c" * 64,
        probe_dir="/probes",
        probe_tree_sha256="d" * 64,
        task_success="0",
        failure_reason="max_steps",
    )
    ledger.append({"event": "attempt_finished", **valid.__dict__})

    assert ledger.valid_results() == {2: valid}


def test_excluded_nodes_are_deduplicated() -> None:
    assert merge_excluded_nodes(("ACD1-8", "ACD1-11"), ("ACD1-8", "")) == (
        "ACD1-11",
        "ACD1-8",
    )


def test_shared_exclusions_propagate_between_tasks() -> None:
    shared = SharedExclusions(("ACD1-8",))
    shared.add("ACD1-29")
    assert shared.snapshot() == ("ACD1-29", "ACD1-8")


def test_launcher_accepts_account_private_log_and_probe_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-' in source
    assert 'PROBE_BASE_DIR="${PROBE_BASE_DIR:-' in source
    assert 'RUNS_ROOT="${RUNS_ROOT:-' in source
    assert "--immediate=120" in source


def test_summary_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_summary(
        run_root / "eval" / "summary.tsv",
        [
            {
                "task_id": "8",
                "episode": "0",
                "seed": "999",
                "task_success": "1",
                "failure_reason": "",
            }
        ],
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps({"hostname": "ACD1-13", "task": 8, "seed": 999, "run_id": "other"}),
        encoding="utf-8",
    )

    result = classify_attempt(
        task=7,
        seed=104,
        attempt=1,
        run_id="run",
        run_root=run_root,
        returncode=0,
        launcher_log=tmp_path / "launcher.log",
        probe_dir=tmp_path / "probes",
    )

    assert result.valid is False
    assert result.failure_reason.startswith("binding_mismatch:")


def test_assignment_requires_every_task_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "assignment.json"
    payload = {
        "schema_version": 1,
        "batch_id": "batch",
        "seed": 104,
        "max_steps": 2500,
        "max_attempts": 4,
        "assignments": {"a": list(range(1, 14)), "b": list(range(14, 27))},
        "slots": {"a": 2, "b": 2},
        "output_roots": {"a": "/a", "b": "/b"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_assignment(path, "a")
    assert loaded["selected_tasks"] == list(range(1, 14))

    payload["assignments"]["b"][-1] = 25
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_assignment(path, "a")


def test_contract_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    ensure_contract(path, {"seed": 104, "max_steps": 2500})

    with pytest.raises(RuntimeError, match="contract mismatch"):
        ensure_contract(path, {"seed": 105, "max_steps": 2500})
