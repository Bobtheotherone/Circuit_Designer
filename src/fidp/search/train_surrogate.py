"""Training utilities for the graph surrogate model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fidp.search.dataset import CircuitDataset, CircuitSample, collate_fn, move_batch_to_device
from fidp.search.surrogate import GraphSurrogateModel, SurrogateConfig
from fidp.search.utils import set_seed


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for surrogate training."""

    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: Optional[int] = 0
    patience: int = 5
    device: Optional[str] = None


@dataclass
class TrainingResult:
    """Result bundle for surrogate training."""

    model: GraphSurrogateModel
    loss_history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")


def _resolve_device(device: Optional[str]) -> "torch.device":
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_surrogate(
    samples: Sequence[CircuitSample],
    surrogate_config: SurrogateConfig,
    feature_config: "FeatureConfig",
    training_config: Optional[TrainingConfig] = None,
) -> TrainingResult:
    """Train the graph surrogate model on a dataset of circuit samples."""
    if not samples:
        raise ValueError("At least one training sample is required.")

    from fidp.search.features import FeatureConfig

    if not isinstance(feature_config, FeatureConfig):
        raise TypeError("feature_config must be a FeatureConfig instance.")

    training_config = training_config or TrainingConfig()
    set_seed(training_config.seed)

    dataset = CircuitDataset(samples=samples, config=feature_config)
    generator = torch.Generator()
    if training_config.seed is not None:
        generator.manual_seed(training_config.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )

    device = _resolve_device(training_config.device)
    model = GraphSurrogateModel(surrogate_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience_left = training_config.patience
    loss_history: List[float] = []

    for _ in range(training_config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(data_loader), 1)
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = training_config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(model=model, loss_history=loss_history, best_loss=best_loss)


def _build_dummy_samples() -> List[CircuitSample]:
    from fidp.search.features import CircuitGraph, Component

    graphs = []
    targets = []
    for idx in range(4):
        comps = [
            Component(kind="R", node_a="1", node_b="0", value=1.0 + idx),
            Component(kind="C", node_a="1", node_b="0", value=1e-6 + idx * 1e-7),
            Component(kind="PORT", node_a="1", node_b="0", value=None),
        ]
        graphs.append(CircuitGraph(components=comps, ports=[]))
        targets.append(np.array([float(idx), float(idx + 1)], dtype=np.float32))
    return [CircuitSample(graph=g, targets=t) for g, t in zip(graphs, targets)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a graph surrogate model.")
    parser.add_argument("--dummy", action="store_true", help="Run a dummy training loop.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for dummy training.")
    args = parser.parse_args()

    if not args.dummy:
        raise SystemExit("Use --dummy to run a small training smoke test.")

    from fidp.search.features import FeatureConfig

    samples = _build_dummy_samples()
    feature_config = FeatureConfig()
    surrogate_config = SurrogateConfig(
        node_feature_dim=feature_config_dim(feature_config),
        output_dim=2,
    )
    training_config = TrainingConfig(epochs=args.epochs, batch_size=2)
    result = train_surrogate(samples, surrogate_config, feature_config, training_config)
    print(f"Training complete. Best loss: {result.best_loss:.4f}")


def feature_config_dim(feature_config: "FeatureConfig") -> int:
    from fidp.search.features import COMPONENT_TYPES

    base = len(COMPONENT_TYPES) + 1  # type one-hot + log value
    if feature_config.include_depth:
        base += 1
    if feature_config.include_distance_to_port:
        base += 1
    return base


if __name__ == "__main__":
    main()
