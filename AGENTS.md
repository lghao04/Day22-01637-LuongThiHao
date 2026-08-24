# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/preference_lab/`, using a `src` package layout. Keep data loading, schemas, losses, trainers, evaluation, configuration, and CLI behavior in their existing focused modules. Tests are in `tests/` and mirror those responsibilities (for example, `data.py` is covered by `test_data.py`). Store small, shareable preference examples in `data/`, experiment settings in `configs/`, supporting material in `docs/`, and utility entry points in `scripts/`. Generated metrics belong in `outputs/`, which should not be committed.

This is an intentionally incomplete teaching repository. Prefer targeted changes to blocks marked `TODO(student)`; avoid broad rewrites unless they are necessary and documented.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local virtual environment.
- `pip install -e '.[dev]'` installs the package, CLI, and development tools. Add `train` (`'.[dev,train]'`) for Torch, Transformers, TRL, and PEFT.
- `make test` or `pytest -q` runs the full test suite.
- `pytest tests/test_data.py` runs one focused test module.
- `make lint` checks `src/` and `tests/` with Ruff.
- `make format` formats Python files with Ruff.
- `make typecheck` runs strict mypy checks against `src/`.
- `make run-eval` evaluates `configs/local.yaml` and writes metrics.

Run lint, type checking, and tests before submitting changes; CI enforces all three on Python 3.11.

## Recommended Lab Workflow

1. **Set up and establish a baseline.** Create and activate `.venv`, install `'.[dev]'`, then run `pytest -q`. On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`; on POSIX shells, use `source .venv/bin/activate`. Record existing failures before editing code.
2. **Inspect and validate the data.** Review `data/sample_preferences.jsonl` and `configs/local.yaml`, then run `pref-lab validate data/sample_preferences.jsonl`. Each JSONL row must contain distinct, non-empty `prompt`, `chosen`, and `rejected` values.
3. **Complete the data path.** Improve validation in `schemas.py` and `data.py` with whitespace-aware comparisons, line-numbered errors, duplicate detection, and deterministic seeded splitting. Keep all rows for the same prompt in one split. Run `pytest tests/test_data.py` after each change.
4. **Optionally expand the dataset.** Install `'.[dev,train]'` only when model training is required. To generate synthetic examples, set `OPENAI_API_KEY` in the environment and run `python scripts/generate_data.py --count 10 --domain "python coding"`. Review generated rows for quality, PII, and duplicates before use; never commit credentials or private data.
5. **Implement one preference objective.** Choose DPO or ORPO and complete the corresponding function in `losses.py` using numerically stable operations. Update TODO-oriented tests to assert computed values and edge cases, then run `pytest tests/test_losses.py`.
6. **Connect training and evaluation.** Implement either the CPU mock or TRL-backed path in `trainers.py`. Replace mock CLI scores with model-derived or deterministic scores, validate score lengths, and define tie behavior in `evaluate.py`. Run `pytest tests/test_evaluate.py`, followed by `pref-lab evaluate --config configs/local.yaml`.
7. **Review outputs and document results.** Inspect `outputs/metrics.json`, compare the safety prompts in `docs/regression_prompts.md` before and after training, update the data card, and complete `docs/REPORT_TEMPLATE.md` with configuration, metrics, qualitative examples, and failure modes. Finish with `make format`, `make lint`, `make typecheck`, and `make test`.

## Coding Style & Naming Conventions

Target Python 3.10+ and use four-space indentation, type annotations, and a 100-character line limit. Follow standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep imports explicit and functions small. Use `pathlib.Path` for filesystem paths and preserve the repository's typed Pydantic-based data boundaries.

## Testing Guidelines

Tests use pytest. Name files `test_<module>.py` and functions `test_<behavior>()`. Add focused tests for successful behavior, validation failures, and edge cases. For dataset splitting, verify examples are grouped by prompt rather than independently by row. Do not require optional training dependencies in ordinary unit tests unless the test is explicitly scoped to training.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects such as `Add report template`; follow that style and keep each commit focused. Pull requests should explain the change, identify affected milestone or issue, and list verification commands and results. Include sample CLI output or metric changes when behavior changes; screenshots are only needed for visual documentation updates. Never commit secrets, model weights, private datasets, caches, or generated artifacts.
