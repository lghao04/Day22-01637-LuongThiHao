import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_rewards_a_better_policy_margin() -> None:
    neutral = dpo_loss([-1.0], [-1.0], [-1.0], [-1.0], beta=1.0)
    preferred = dpo_loss([-0.5], [-1.5], [-1.0], [-1.0], beta=1.0)

    assert neutral == pytest.approx(np.log(2.0))
    assert preferred < neutral


def test_dpo_loss_is_stable_for_extreme_logits() -> None:
    loss = dpo_loss([1e6], [-1e6], [0.0], [0.0], beta=1.0)
    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0)


def test_dpo_loss_validates_shapes_and_beta() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dpo_loss([-0.5, -0.2], [-1.5], [-0.6], [-1.0], beta=0.1)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss([-0.5], [-1.5], [-0.6], [-1.0], beta=0.0)


def test_orpo_loss_combines_sft_and_preference_penalty() -> None:
    loss = orpo_loss([1.0], [-0.5], [-1.5], lambda_orpo=0.1)

    assert np.isfinite(loss)
    assert loss > 1.0


def test_orpo_zero_weight_equals_mean_sft_loss() -> None:
    loss = orpo_loss([1.0, 3.0], [-0.5, -0.7], [-1.5, -1.2], lambda_orpo=0.0)
    assert loss == pytest.approx(2.0)


def test_orpo_rejects_positive_log_probabilities() -> None:
    with pytest.raises(ValueError, match="less than or equal to zero"):
        orpo_loss([1.0], [0.1], [-1.5], lambda_orpo=0.1)
