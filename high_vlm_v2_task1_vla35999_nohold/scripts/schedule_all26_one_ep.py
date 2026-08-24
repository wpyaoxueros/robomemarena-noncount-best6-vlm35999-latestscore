#!/usr/bin/env python3
"""Run disjoint Task1-26 evaluation assignments with auditable retries."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDED_NODES = ("ACD1-8", "ACD1-11", "ACD1-31", "ACD1-54")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_tasks(value: str) -> tuple[int, ...]:
    tasks: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"descending task range is not allowed: {item}")
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(item))
    invalid = sorted(task for task in tasks if task < 1 or task > 26)
    if invalid or not tasks:
        raise ValueError(f"tasks must be a non-empty subset of 1..26; invalid={invalid}")
    return tuple(sorted(tasks))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(frozen=True)
class AttemptResult:
    task: int
    attempt: int
    run_id: str
    valid: bool
    returncode: int
    node: str | None
    summary_path: str | None
    summary_sha256: str | None
    manifest_path: str | None
    manifest_sha256: str | None
    launcher_log_path: str | None
    launcher_log_sha256: str | None
    probe_dir: str | None
    probe_tree_sha256: str | None
    task_success: str | None
    failure_reason: str


def classify_attempt(
    *,
    task: int,
    seed: int,
    attempt: int,
    run_id: str,
    run_root: Path,
    returncode: int,
    launcher_log: Path,
    probe_dir: Path,
) -> AttemptResult:
    summary_path = run_root / "eval" / "summary.tsv"
    manifest_path = run_root / "run_manifest.json"
    allocation_identity_path = run_root / "allocation_identity.json"
    rows = read_summary_rows(summary_path)
    manifest = read_manifest(manifest_path)
    allocation_identity = read_manifest(allocation_identity_path)
    node = manifest.get("hostname") or allocation_identity.get("hostname")
    node = str(node) if node else None

    launcher_log_path = str(launcher_log.resolve()) if launcher_log.is_file() else None
    launcher_log_sha256 = sha256(launcher_log) if launcher_log.is_file() else None
    probe_dir_path = str(probe_dir.resolve()) if probe_dir.is_dir() else None
    probe_tree_sha256 = hash_tree(probe_dir) if probe_dir.is_dir() else None

    if len(rows) == 1:
        row = rows[0]
        bindings = {
            "summary.task_id": row.get("task_id") == str(task),
            "summary.seed": row.get("seed") == str(seed),
            "summary.episode": row.get("episode") == "0",
            "manifest.task": manifest.get("task") == task,
            "manifest.seed": manifest.get("seed") == seed,
            "manifest.run_id": manifest.get("run_id") == run_id,
            "summary.task_success": row.get("task_success") in {"0", "1"},
        }
        mismatches = [name for name, matches in bindings.items() if not matches]
        if mismatches:
            return AttemptResult(
                task=task,
                attempt=attempt,
                run_id=run_id,
                valid=False,
                returncode=returncode,
                node=node,
                summary_path=str(summary_path.resolve()),
                summary_sha256=sha256(summary_path),
                manifest_path=str(manifest_path.resolve()) if manifest_path.is_file() else None,
                manifest_sha256=sha256(manifest_path) if manifest_path.is_file() else None,
                launcher_log_path=launcher_log_path,
                launcher_log_sha256=launcher_log_sha256,
                probe_dir=probe_dir_path,
                probe_tree_sha256=probe_tree_sha256,
                task_success=None,
                failure_reason="binding_mismatch:" + ",".join(mismatches),
            )
        return AttemptResult(
            task=task,
            attempt=attempt,
            run_id=run_id,
            valid=True,
            returncode=returncode,
            node=node,
            summary_path=str(summary_path.resolve()),
            summary_sha256=sha256(summary_path),
            manifest_path=str(manifest_path.resolve()) if manifest_path.is_file() else None,
            manifest_sha256=sha256(manifest_path) if manifest_path.is_file() else None,
            launcher_log_path=launcher_log_path,
            launcher_log_sha256=launcher_log_sha256,
            probe_dir=probe_dir_path,
            probe_tree_sha256=probe_tree_sha256,
            task_success=row.get("task_success"),
            failure_reason=row.get("failure_reason", "") or "valid_summary",
        )

    if len(rows) > 1:
        reason = f"invalid_summary_row_count:{len(rows)}"
    elif returncode != 0:
        reason = f"launcher_exit:{returncode}:no_summary_row"
    else:
        reason = "launcher_exit:0:no_summary_row"
    return AttemptResult(
        task=task,
        attempt=attempt,
        run_id=run_id,
        valid=False,
        returncode=returncode,
        node=node,
        summary_path=str(summary_path.resolve()) if summary_path.is_file() else None,
        summary_sha256=sha256(summary_path) if summary_path.is_file() else None,
        manifest_path=str(manifest_path.resolve()) if manifest_path.is_file() else None,
        manifest_sha256=sha256(manifest_path) if manifest_path.is_file() else None,
        launcher_log_path=launcher_log_path,
        launcher_log_sha256=launcher_log_sha256,
        probe_dir=probe_dir_path,
        probe_tree_sha256=probe_tree_sha256,
        task_success=None,
        failure_reason=reason,
    )


class EventLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: dict[str, object]) -> None:
        payload = {"utc": utc_now(), **event}
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

    def valid_results(self) -> dict[int, AttemptResult]:
        if not self.path.is_file():
            return {}
        valid: dict[int, AttemptResult] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "attempt_finished" and payload.get("valid") is True:
                result_fields = {
                    field: payload.get(field) for field in AttemptResult.__dataclass_fields__
                }
                valid[int(payload["task"])] = AttemptResult(**result_fields)
        return valid


class SharedExclusions:
    """Share newly observed bad nodes across concurrent tasks for one account."""

    def __init__(self, initial: Iterable[str]) -> None:
        self._nodes = {node.strip() for node in initial if node.strip()}
        self._lock = threading.Lock()

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._nodes))

    def add(self, node: str) -> None:
        with self._lock:
            self._nodes.add(node.strip())


def hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_contract(path: Path, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    expected_sha256 = canonical_sha256(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"scheduler contract mismatch: {path}")
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return expected_sha256


def load_assignment(path: Path, expected_user: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("assignment schema_version must be 1")
    assignments = payload.get("assignments")
    slots = payload.get("slots")
    output_roots = payload.get("output_roots")
    if not isinstance(assignments, dict) or not isinstance(slots, dict) or not isinstance(output_roots, dict):
        raise ValueError("assignment must define assignments, slots, and output_roots mappings")
    flattened = [int(task) for tasks in assignments.values() for task in tasks]
    if len(flattened) != 26 or sorted(flattened) != list(range(1, 27)):
        raise ValueError("assignment must contain every Task1-26 exactly once")
    if expected_user not in assignments or expected_user not in slots or expected_user not in output_roots:
        raise ValueError(f"assignment has no complete entry for {expected_user}")
    payload["selected_tasks"] = list(parse_tasks(",".join(str(task) for task in assignments[expected_user])))
    payload["selected_slots"] = int(slots[expected_user])
    payload["selected_output_root"] = str(output_roots[expected_user])
    return payload


def merge_excluded_nodes(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({node.strip() for group in groups for node in group if node.strip()}))


def write_status(path: Path, results: dict[int, AttemptResult], assigned: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "task",
                "state",
                "attempt",
                "run_id",
                "task_success",
                "node",
                "failure_reason",
                "summary_path",
            ]
        )
        for task in assigned:
            result = results.get(task)
            if result is None:
                writer.writerow([task, "pending", "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [
                        task,
                        "valid" if result.valid else "blocked",
                        result.attempt,
                        result.run_id,
                        result.task_success or "",
                        result.node or "",
                        result.failure_reason,
                        result.summary_path or "",
                    ]
                )
    temporary.replace(path)


def run_task(
    *,
    task: int,
    args: argparse.Namespace,
    exp_root: Path,
    runs_root: Path,
    ledger: EventLedger,
    contract_sha256: str,
    shared_exclusions: SharedExclusions,
) -> AttemptResult:
    last_result: AttemptResult | None = None
    for attempt in range(1, args.max_attempts + 1):
        excluded = shared_exclusions.snapshot()
        run_id = (
            f"{args.batch_id}_task{task:02d}_seed{args.seed}_"
            f"a{attempt}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_root = runs_root / run_id
        launcher_log = args.state_dir / "launcher_logs" / f"{run_id}.log"
        probe_dir = args.state_dir / "probes" / run_id
        env = os.environ.copy()
        env.update(
            {
                "TASK_ID": str(task),
                "SEED": str(args.seed),
                "MAX_STEPS": str(args.max_steps),
                "EXPECTED_UNIX_USER": args.expected_user,
                "RUN_ID": run_id,
                "EXCLUDE_NODES": ",".join(excluded),
                "LAUNCH_LOG_DIR": str(args.state_dir / "launcher_logs"),
                "PROBE_BASE_DIR": str(args.state_dir / "probes"),
                "RUNS_ROOT": str(runs_root),
            }
        )
        ledger.append(
            {
                "event": "attempt_started",
                "task": task,
                "attempt": attempt,
                "run_id": run_id,
                "contract_sha256": contract_sha256,
                "excluded_nodes": list(excluded),
            }
        )
        completed = subprocess.run(
            [str(args.launcher)],
            cwd=exp_root.parent,
            env=env,
            check=False,
        )
        result = classify_attempt(
            task=task,
            seed=args.seed,
            attempt=attempt,
            run_id=run_id,
            run_root=run_root,
            returncode=completed.returncode,
            launcher_log=launcher_log,
            probe_dir=probe_dir,
        )
        ledger.append(
            {"event": "attempt_finished", "contract_sha256": contract_sha256, **asdict(result)}
        )
        last_result = result
        if result.valid:
            return result
        if result.node:
            shared_exclusions.add(result.node)

    assert last_result is not None
    return last_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-user", required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    actual = subprocess.check_output(["whoami"], text=True).strip()
    if actual != args.expected_user:
        raise SystemExit(f"expected user {args.expected_user}, got {actual}")

    exp_root = args.exp_root.resolve()
    assignment_path = args.assignment_manifest.resolve()
    assignment = load_assignment(assignment_path, args.expected_user)
    assigned = tuple(int(task) for task in assignment["selected_tasks"])
    args.slots = int(assignment["selected_slots"])
    args.batch_id = str(assignment["batch_id"])
    args.seed = int(assignment["seed"])
    args.max_steps = int(assignment["max_steps"])
    args.max_attempts = int(assignment["max_attempts"])
    args.exclude_nodes = ",".join(str(node) for node in assignment.get("exclude_nodes", []))
    if args.slots < 1 or args.max_attempts < 1:
        raise SystemExit("assignment slots and max_attempts must be at least 1")
    output_root = Path(str(assignment["selected_output_root"])).resolve()
    runs_root = output_root / "runs"
    args.launcher = exp_root / "scripts" / "launch_all26_one_ep_account.sh"
    args.state_dir = output_root / "scheduler_state"
    if not args.launcher.is_file():
        raise SystemExit(f"missing launcher: {args.launcher}")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.state_dir / "scheduler.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another scheduler owns {args.state_dir}") from exc

    scheduler_path = Path(__file__).resolve()
    contract = {
        "schema_version": 1,
        "batch_id": args.batch_id,
        "expected_user": args.expected_user,
        "assigned_tasks": list(assigned),
        "slots": args.slots,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_attempts": args.max_attempts,
        "exclude_nodes": args.exclude_nodes,
        "assignment_manifest": str(assignment_path),
        "assignment_manifest_sha256": sha256(assignment_path),
        "scheduler": str(scheduler_path),
        "scheduler_sha256": sha256(scheduler_path),
        "launcher": str(args.launcher),
        "launcher_sha256": sha256(args.launcher),
        "output_root": str(output_root),
        "runs_root": str(runs_root),
    }
    contract_sha256 = ensure_contract(args.state_dir / "contract.json", contract)
    ledger = EventLedger(args.state_dir / "events.jsonl")
    results = ledger.valid_results()
    unexpected = sorted(set(results) - set(assigned))
    if unexpected:
        raise RuntimeError(f"state directory contains unassigned valid tasks: {unexpected}")
    pending = tuple(task for task in assigned if task not in results)
    shared_exclusions = SharedExclusions(
        merge_excluded_nodes(DEFAULT_EXCLUDED_NODES, args.exclude_nodes.split(","))
    )
    ledger.append(
        {
            "event": "scheduler_started",
            "batch_id": args.batch_id,
            "expected_user": args.expected_user,
            "hostname": socket.gethostname(),
            "assigned_tasks": list(assigned),
            "pending_tasks": list(pending),
            "slots": args.slots,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "max_attempts": args.max_attempts,
            "contract_sha256": contract_sha256,
            "scheduler": str(scheduler_path),
            "scheduler_sha256": sha256(scheduler_path),
            "launcher": str(args.launcher),
            "launcher_sha256": sha256(args.launcher),
            "assignment_manifest": str(assignment_path),
            "assignment_manifest_sha256": sha256(assignment_path),
            "output_root": str(output_root),
            "runs_root": str(runs_root),
        }
    )

    status_path = args.state_dir / "status.tsv"
    write_status(status_path, results, assigned)
    with ThreadPoolExecutor(max_workers=min(args.slots, len(pending) or 1)) as executor:
        futures = {
            executor.submit(
                run_task,
                task=task,
                args=args,
                exp_root=exp_root,
                runs_root=runs_root,
                ledger=ledger,
                contract_sha256=contract_sha256,
                shared_exclusions=shared_exclusions,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                results[task] = future.result()
            except Exception as exc:  # Preserve scheduler errors in the append-only ledger.
                ledger.append(
                    {
                        "event": "scheduler_task_exception",
                        "task": task,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                raise
            finally:
                write_status(status_path, results, assigned)

    blocked = sorted(task for task, result in results.items() if not result.valid)
    ledger.append(
        {
            "event": "scheduler_finished",
            "contract_sha256": contract_sha256,
            "valid_tasks": sorted(task for task, result in results.items() if result.valid),
            "blocked_tasks": blocked,
        }
    )
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
