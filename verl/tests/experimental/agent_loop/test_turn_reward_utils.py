import pytest

from verl.experimental.agent_loop.turn_reward_utils import build_trace_layout


def test_trace_layout_uses_pre_generation_states_and_excludes_answer_tail():
    layout = build_trace_layout(
        assistant_turn_spans=[(0, 2), (4, 7), (9, 11)],
        response_length=11,
    )

    assert layout.state_boundaries == [0, 4, 9]
    assert layout.turn_spans == [(0, 2), (4, 7)]
    assert layout.final_answer_span == (9, 11)


def test_trace_layout_clips_the_final_span_without_inventing_a_transition():
    layout = build_trace_layout(
        assistant_turn_spans=[(0, 3), (5, 9)],
        response_length=7,
    )

    assert layout.state_boundaries == [0, 5]
    assert layout.turn_spans == [(0, 3)]
    assert layout.final_answer_span == (5, 7)


def test_trace_layout_validates_monotonic_spans():
    with pytest.raises(ValueError, match="overlap"):
        build_trace_layout([(0, 3), (2, 4)], response_length=4)
