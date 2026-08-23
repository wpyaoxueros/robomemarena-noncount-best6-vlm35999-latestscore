from __future__ import annotations

from types import SimpleNamespace

from extensions.eval_two_call_prompt_commit import TwoCallPromptCommitter, install_extension


ACTIVE = "pick cookies"
NEXT = "place cookies into basket"
OTHER = "pick tomato sauce"


def make_committer() -> TwoCallPromptCommitter:
    return TwoCallPromptCommitter(required_calls=2, initial_prompt=ACTIVE)


def test_first_new_prediction_stays_on_active_prompt() -> None:
    decision = make_committer().observe(NEXT)

    assert decision.active_prompt == ACTIVE
    assert decision.candidate_prompt == NEXT
    assert decision.candidate_count == 1
    assert decision.committed is False


def test_second_consecutive_prediction_commits_new_prompt() -> None:
    committer = make_committer()
    committer.observe(NEXT)

    decision = committer.observe(NEXT)

    assert decision.active_prompt == NEXT
    assert decision.candidate_prompt == ""
    assert decision.candidate_count == 0
    assert decision.committed is True


def test_different_candidate_replaces_pending_candidate() -> None:
    committer = make_committer()
    committer.observe(NEXT)

    decision = committer.observe(OTHER)

    assert decision.active_prompt == ACTIVE
    assert decision.candidate_prompt == OTHER
    assert decision.candidate_count == 1
    assert decision.committed is False


def test_active_prompt_clears_pending_candidate() -> None:
    committer = make_committer()
    committer.observe(NEXT)

    decision = committer.observe(ACTIVE)

    assert decision.active_prompt == ACTIVE
    assert decision.candidate_prompt == ""
    assert decision.candidate_count == 0
    assert decision.committed is False


def test_empty_or_error_output_clears_candidate_without_switching() -> None:
    committer = make_committer()
    committer.observe(NEXT)

    empty = committer.observe("")
    assert empty.active_prompt == ACTIVE
    assert empty.candidate_prompt == ""
    assert empty.candidate_count == 0

    committer.observe(NEXT)
    error = committer.observe(NEXT, inference_error=True)
    assert error.active_prompt == ACTIVE
    assert error.candidate_prompt == ""
    assert error.candidate_count == 0
    assert error.committed is False


def test_reset_restores_initial_prompt_and_clears_candidate() -> None:
    committer = make_committer()
    committer.observe(NEXT)
    committer.observe(NEXT)
    assert committer.active_prompt == NEXT

    committer.reset(ACTIVE)

    assert committer.active_prompt == ACTIVE
    assert committer.candidate_prompt == ""
    assert committer.candidate_count == 0


def test_planner_wrapper_counts_fresh_parsed_predictions_only() -> None:
    class FakeBasePlanner:
        default_subtask_prompt = ACTIVE

        def __init__(self, outputs: list[tuple[str, str]]) -> None:
            self.outputs = iter(outputs)

        def reset_episode(self) -> None:
            return None

        def infer_record(self, _request: object) -> SimpleNamespace:
            parsed_prompt, persistent_prompt = next(self.outputs)
            return SimpleNamespace(
                prompt=persistent_prompt,
                event={"parsed_primitive": parsed_prompt},
                error=None,
            )

    runner = SimpleNamespace(_make_recording_planner_class=lambda _ref: FakeBasePlanner)
    install_extension(runner, required_calls=2)
    planner_class = runner._make_recording_planner_class(None)
    planner = planner_class(
        [
            (NEXT, NEXT),
            ("", NEXT),
            (NEXT, NEXT),
            (NEXT, NEXT),
        ]
    )
    planner.reset_episode()

    first = planner.infer_record(object())
    assert first.prompt == ACTIVE
    assert first.event["controller_candidate_count"] == 1

    parse_failure = planner.infer_record(object())
    assert parse_failure.prompt == ACTIVE
    assert parse_failure.event["model_accepted_prompt"] == NEXT
    assert parse_failure.event["raw_controller_prompt"] == ""
    assert parse_failure.event["controller_candidate_count"] == 0

    third = planner.infer_record(object())
    assert third.prompt == ACTIVE
    assert third.event["controller_candidate_count"] == 1

    fourth = planner.infer_record(object())
    assert fourth.prompt == NEXT
    assert fourth.event["controller_prompt_committed"] is True
    assert fourth.event["accepted_controller_prompt"] == NEXT
