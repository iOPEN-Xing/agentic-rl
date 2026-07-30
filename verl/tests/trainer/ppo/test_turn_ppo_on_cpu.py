from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_turn_ppo,
    compute_turn_gae_advantage_return,
)


def _actor_config(clip_ratio: float = 0.2):
    return SimpleNamespace(
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
    )


def test_turn_gae_propagates_terminal_reward_over_turns_not_tokens():
    response_mask = torch.tensor([[1, 1, 0, 1, 1, 0]], dtype=torch.float32)
    turn_ids = torch.tensor([[1, 1, 0, 2, 2, 0]], dtype=torch.long)
    values = torch.zeros_like(response_mask)
    token_rewards = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.float32)

    advantages, returns, turn_value_mask = compute_turn_gae_advantage_return(
        token_level_rewards=token_rewards,
        values=values,
        response_mask=response_mask,
        turn_ids=turn_ids,
        gamma=0.5,
        lam=1.0,
    )

    assert advantages[0, 0].item() == pytest.approx(advantages[0, 1].item())
    assert advantages[0, 3].item() == pytest.approx(advantages[0, 4].item())
    assert advantages[0, 3].item() > advantages[0, 0].item()
    assert torch.equal(advantages[response_mask == 0], torch.zeros(2))
    assert returns[0, 0].item() == pytest.approx(0.5)
    assert returns[0, 3].item() == pytest.approx(1.0)
    assert torch.equal(
        turn_value_mask,
        torch.tensor([[0.5, 0, 0, 0.5, 0, 0]], dtype=torch.float32),
    )


def test_turn_gae_bootstraps_from_the_next_turn_state_value():
    response_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    turn_ids = torch.tensor([[1, 0, 2, 0]], dtype=torch.long)
    values = torch.tensor([[0.2, 0, 0.4, 0]], dtype=torch.float32)
    token_rewards = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)

    _, returns, _ = compute_turn_gae_advantage_return(
        token_level_rewards=token_rewards,
        values=values,
        response_mask=response_mask,
        turn_ids=turn_ids,
        gamma=1.0,
        lam=1.0,
    )

    assert returns[0, 0].item() == pytest.approx(1.0)
    assert returns[0, 2].item() == pytest.approx(1.0)


def test_turn_gae_rejects_non_contiguous_turn_ids():
    response_mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    turn_ids = torch.tensor([[1, 0, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="contiguous"):
        compute_turn_gae_advantage_return(
            token_level_rewards=torch.zeros_like(response_mask),
            values=torch.zeros_like(response_mask),
            response_mask=response_mask,
            turn_ids=turn_ids,
            gamma=1.0,
            lam=1.0,
        )


def test_turn_gae_rejects_repeated_or_split_turn_spans():
    response_mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    with pytest.raises(ValueError, match="contiguous assistant span"):
        compute_turn_gae_advantage_return(
            token_level_rewards=torch.zeros_like(response_mask),
            values=torch.zeros_like(response_mask),
            response_mask=response_mask,
            turn_ids=torch.tensor([[1, 0, 1]]),
            gamma=1.0,
            lam=1.0,
        )

    reordered_response_mask = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32)
    with pytest.raises(ValueError, match="chronological"):
        compute_turn_gae_advantage_return(
            token_level_rewards=torch.zeros_like(reordered_response_mask),
            values=torch.zeros_like(reordered_response_mask),
            response_mask=reordered_response_mask,
            turn_ids=torch.tensor([[1, 0, 2, 0, 1]]),
            gamma=1.0,
            lam=1.0,
        )


def test_turn_policy_loss_normalizes_by_trajectory_assistant_tokens():
    old_log_prob = torch.zeros((1, 4))
    log_prob = torch.zeros((1, 4), requires_grad=True)
    response_mask = torch.ones((1, 4))
    turn_ids = torch.tensor([[1, 2, 2, 2]])
    advantages = torch.tensor([[1.0, 3.0, 3.0, 3.0]])

    loss, metrics = compute_policy_loss_turn_ppo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        turn_ids=turn_ids,
        config=_actor_config(),
    )

    # Table 1: (-1 + -3) / four assistant tokens.
    assert loss.item() == pytest.approx(-1.0)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)


def test_turn_policy_loss_averages_trajectories_not_global_tokens():
    old_log_prob = torch.zeros((2, 4))
    log_prob = torch.zeros((2, 4), requires_grad=True)
    response_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    turn_ids = torch.tensor([[1, 0, 0, 0], [1, 2, 2, 2]])
    advantages = torch.tensor([[2.0, 0, 0, 0], [1.0, 3.0, 3.0, 3.0]])

    loss, _ = compute_policy_loss_turn_ppo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        turn_ids=turn_ids,
        config=_actor_config(),
    )

    # Trajectory objectives are -2/1 and (-1 + -3)/4, then batch-meaned.
    assert loss.item() == pytest.approx(-1.5)


def test_turn_policy_loss_uses_product_ratio_and_clips_the_complete_turn():
    old_log_prob = torch.zeros((1, 2))
    log_prob = torch.tensor([[0.2, 0.2]], requires_grad=True)
    response_mask = torch.ones((1, 2))
    turn_ids = torch.tensor([[1, 1]])
    advantages = torch.ones((1, 2))

    loss, metrics = compute_policy_loss_turn_ppo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        turn_ids=turn_ids,
        config=_actor_config(clip_ratio=0.2),
    )

    # One clipped turn objective divided by two assistant tokens.
    assert loss.item() == pytest.approx(-0.6)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)


def test_turn_policy_loss_rejects_active_tokens_without_turn_identity():
    with pytest.raises(ValueError, match="positive turn id"):
        compute_policy_loss_turn_ppo(
            old_log_prob=torch.zeros((1, 2)),
            log_prob=torch.zeros((1, 2)),
            advantages=torch.ones((1, 2)),
            response_mask=torch.ones((1, 2)),
            turn_ids=torch.tensor([[1, 0]]),
            config=_actor_config(),
        )


def test_turn_policy_loss_rejects_additional_rollout_is_weights():
    with pytest.raises(ValueError, match="does not define an additional"):
        compute_policy_loss_turn_ppo(
            old_log_prob=torch.zeros((1, 1)),
            log_prob=torch.zeros((1, 1)),
            advantages=torch.ones((1, 1)),
            response_mask=torch.ones((1, 1)),
            turn_ids=torch.ones((1, 1), dtype=torch.long),
            rollout_is_weights=torch.ones((1, 1)),
            config=_actor_config(),
        )
