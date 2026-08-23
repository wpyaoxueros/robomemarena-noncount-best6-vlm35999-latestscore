#!/usr/bin/env python3
"""Apply two-call confirmation before VLM prompts control the VLA."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXP_ROOT = Path(os.environ.get("HV2_EXP_ROOT", Path(__file__).resolve().parents[1])).resolve()
SOURCE_VLM_DIR = EXP_ROOT / "source/vlm_ft"


@dataclass(frozen=True)
class PromptCommitDecision:
    active_prompt: str
    candidate_prompt: str
    candidate_count: int
    committed: bool


class TwoCallPromptCommitter:
    """Require repeated fresh predictions before changing controller prompt."""

    def __init__(self, *, required_calls: int, initial_prompt: str) -> None:
        if int(required_calls) < 1:
            raise ValueError("required_calls must be a positive integer")
        self.required_calls = int(required_calls)
        self.active_prompt = ""
        self.candidate_prompt = ""
        self.candidate_count = 0
        self.reset(initial_prompt)

    def reset(self, initial_prompt: str) -> None:
        prompt = str(initial_prompt).strip()
        if not prompt:
            raise ValueError("initial_prompt must be non-empty")
        self.active_prompt = prompt
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self.candidate_prompt = ""
        self.candidate_count = 0

    def observe(
        self,
        raw_prompt: str,
        *,
        inference_error: bool = False,
    ) -> PromptCommitDecision:
        prompt = str(raw_prompt).strip()
        committed = False

        if inference_error or not prompt or prompt == self.active_prompt:
            self._clear_candidate()
        elif prompt == self.candidate_prompt:
            self.candidate_count += 1
        else:
            self.candidate_prompt = prompt
            self.candidate_count = 1

        if self.candidate_count >= self.required_calls:
            self.active_prompt = self.candidate_prompt
            self._clear_candidate()
            committed = True

        return PromptCommitDecision(
            active_prompt=self.active_prompt,
            candidate_prompt=self.candidate_prompt,
            candidate_count=self.candidate_count,
            committed=committed,
        )


def install_extension(runner: Any, *, required_calls: int = 2) -> None:
    if int(required_calls) < 1:
        raise ValueError("required_calls must be a positive integer")
    original_factory = runner._make_recording_planner_class  # noqa: SLF001

    def make_recording_planner_class(ref: Any) -> type[Any]:
        base_planner = original_factory(ref)

        class ConfirmedRecordingPlanner(base_planner):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._controller_prompt_committer = TwoCallPromptCommitter(
                    required_calls=int(required_calls),
                    initial_prompt=self.default_subtask_prompt,
                )

            def reset_episode(self, *args: Any, **kwargs: Any) -> Any:
                result = super().reset_episode(*args, **kwargs)
                self._controller_prompt_committer.reset(self.default_subtask_prompt)
                return result

            def infer_record(self, request: Any) -> Any:
                result = super().infer_record(request)
                model_accepted_prompt = str(result.prompt or "")
                raw_prompt = str(result.event.get("parsed_primitive") or "")
                decision = self._controller_prompt_committer.observe(
                    raw_prompt,
                    inference_error=bool(result.error),
                )
                result.event.update(
                    {
                        "raw_controller_prompt": raw_prompt,
                        "model_accepted_prompt": model_accepted_prompt,
                        "controller_candidate_prompt": decision.candidate_prompt,
                        "controller_candidate_count": decision.candidate_count,
                        "controller_required_calls": int(required_calls),
                        "controller_prompt_committed": decision.committed,
                        "accepted_controller_prompt": decision.active_prompt,
                    }
                )
                result.prompt = decision.active_prompt
                return result

        ConfirmedRecordingPlanner.__name__ = (
            f"Confirmed{int(required_calls)}Call{base_planner.__name__}"
        )
        return ConfirmedRecordingPlanner

    runner._make_recording_planner_class = make_recording_planner_class  # noqa: SLF001


def _required_calls_from_env() -> int:
    raw = os.environ.get("CONTROLLER_STABLE_VLM_CALLS", "2")
    try:
        required_calls = int(raw)
    except ValueError as exc:
        raise ValueError(f"CONTROLLER_STABLE_VLM_CALLS must be an integer, got {raw!r}") from exc
    if required_calls < 1:
        raise ValueError("CONTROLLER_STABLE_VLM_CALLS must be positive")
    return required_calls


def main() -> None:
    if str(SOURCE_VLM_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_VLM_DIR))
    import eval_three_tasks as runner

    install_extension(runner, required_calls=_required_calls_from_env())
    runner.main()


if __name__ == "__main__":
    main()
