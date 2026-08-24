from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import deterministic_score, pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


def _build_training_config(config: dict[str, object]) -> TrainingConfig:
    raw_training = config.get("training", {})
    raw_paths = config.get("paths", {})
    if not isinstance(raw_training, dict) or not isinstance(raw_paths, dict):
        raise TypeError("paths and training configuration must be mappings")
    return TrainingConfig(
        method=str(raw_training.get("method", "mock")),
        beta=float(raw_training.get("beta", 0.1)),
        lambda_orpo=float(raw_training.get("lambda_orpo", 0.1)),
        max_length=int(raw_training.get("max_length", 512)),
        batch_size=int(raw_training.get("batch_size", 2)),
        output_dir=str(raw_paths.get("output_dir", "outputs")),
    )


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", help="Path to the YAML config.")],
) -> None:
    """Run the deterministic CPU preference-training pipeline."""
    cfg = load_config(config)
    training_config = _build_training_config(cfg)
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or "train_data" not in paths:
        raise ValueError("configuration must define paths.train_data")
    examples = load_jsonl(str(paths["train_data"]))
    metrics = PreferenceTrainer(training_config).train(examples)
    print(f"[green]Training complete: loss={metrics['final_loss']:.4f}[/green]")


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="Path to the YAML config.")],
) -> None:
    cfg = load_config(config)
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or "train_data" not in paths or "output_dir" not in paths:
        raise ValueError("configuration must define paths.train_data and paths.output_dir")
    examples = load_jsonl(str(paths["train_data"]))
    chosen_scores = [deterministic_score(example.chosen) for example in examples]
    rejected_scores = [deterministic_score(example.rejected) for example in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, str(paths["output_dir"]))
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
