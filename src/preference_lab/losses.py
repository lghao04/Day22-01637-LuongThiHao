from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _float_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_same_shape(arrays: list[NDArray[np.float64]]) -> None:
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all loss inputs must have the same shape")


def _log_one_minus_exp(log_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    cutoff = -np.log(2.0)
    return np.where(
        log_probabilities < cutoff,
        np.log1p(-np.exp(log_probabilities)),
        np.log(-np.expm1(log_probabilities)),
    )


def dpo_loss(
    policy_chosen_logps: ArrayLike,
    policy_rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    beta: float,
) -> float:
    """Compute the mean, numerically stable DPO loss for a batch."""
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be a positive finite number")

    arrays = [
        _float_array("policy_chosen_logps", policy_chosen_logps),
        _float_array("policy_rejected_logps", policy_rejected_logps),
        _float_array("ref_chosen_logps", ref_chosen_logps),
        _float_array("ref_rejected_logps", ref_rejected_logps),
    ]
    _require_same_shape(arrays)
    policy_log_ratio = arrays[0] - arrays[1]
    reference_log_ratio = arrays[2] - arrays[3]
    logits = beta * (policy_log_ratio - reference_log_ratio)
    return float(np.mean(np.logaddexp(0.0, -logits)))


def orpo_loss(
    sft_nll: ArrayLike,
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    lambda_orpo: float,
) -> float:
    """Compute mean SFT NLL plus a stable odds-ratio preference penalty."""
    if lambda_orpo < 0 or not np.isfinite(lambda_orpo):
        raise ValueError("lambda_orpo must be a non-negative finite number")

    arrays = [
        _float_array("sft_nll", sft_nll),
        _float_array("chosen_logps", chosen_logps),
        _float_array("rejected_logps", rejected_logps),
    ]
    _require_same_shape(arrays)
    if np.any(arrays[0] < 0):
        raise ValueError("sft_nll must be non-negative")
    if np.any(arrays[1] > 0) or np.any(arrays[2] > 0):
        raise ValueError("log probabilities must be less than or equal to zero")

    upper_bound = -np.finfo(np.float64).eps
    chosen = np.minimum(arrays[1], upper_bound)
    rejected = np.minimum(arrays[2], upper_bound)
    chosen_log_odds = chosen - _log_one_minus_exp(chosen)
    rejected_log_odds = rejected - _log_one_minus_exp(rejected)
    preference_penalty = np.logaddexp(0.0, -(chosen_log_odds - rejected_log_odds))
    return float(np.mean(arrays[0]) + lambda_orpo * np.mean(preference_penalty))
