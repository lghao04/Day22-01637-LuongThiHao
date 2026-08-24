# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary

- **Total examples loaded:** 24
- **Validation issues found:** Line 1 contained unescaped quotation marks around `self-attention` and was invalid JSON.
- **Cleaning steps taken:** Escaped the quotation marks and added line-numbered JSON/schema errors, normalized duplicate-prompt detection, and optional email/phone checks.

### Split Strategy

- **Train/Val Ratio:** 19/5 (approximately 80/20), seed 42.
- **Leakage Prevention:** Prompts are normalized and grouped before deterministic shuffling, so variants of one prompt cannot occur in both splits.

## 2. Implementation: DPO

### Objective Selection

- **Why this method?** DPO directly optimizes chosen-versus-rejected margins and does not require a separate reward model.
- **Key Hyperparameters:** `beta=0.1`, `max_length=512`, and `batch_size=2`.

### Numerical Stability

- **Challenges:** Large preference logits can overflow a direct sigmoid/log calculation.
- **Solutions:** The implementation uses `numpy.logaddexp` for stable negative log-sigmoid evaluation and rejects non-finite or shape-mismatched inputs. ORPO is also implemented with stable `log1p`/`expm1` odds calculations.

## 3. Evaluation Results

### Metrics

| Metric | Value |
|---|---|
| Pairwise Accuracy | `100%` |
| Final Loss (CPU deterministic DPO) | `0.692662` |

### Qualitative Review

- **Prompt:** Explain the concept of "self-attention" in Transformers.
- **Chosen Response:** Explains weighting input words and capturing long-range dependencies.
- **Rejected Response:** Incorrectly describes self-attention as a simpler RNN.
- **Model Preference:** Correct under the deterministic lexical scorer.

## 4. Discussion & Failure Modes

- **What went well?** All 24 examples validate, the prompt-grouped split has no leakage, and unit tests cover invalid data, duplicate prompts, stable losses, ties, and trainer output.
- **Observed Bias:** The CPU scorer uses response length and lexical diversity. The dataset's chosen responses are usually longer, so 100% accuracy does not demonstrate semantic understanding or model improvement.
- **Safety:** The regression prompts were reviewed, but before/after response safety cannot be measured because this run uses a non-generative CPU scorer. A TRL/model-backed run is required before making safety claims.
