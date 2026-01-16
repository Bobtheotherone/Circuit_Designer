"""Active learning loop orchestrating evaluation, training, and proposal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from fidp.search.botorch_mobo import is_botorch_available, propose_next_botorch, propose_next_random
from fidp.search.dataset import CircuitSample
from fidp.search.evolution import DesignRecord, EvolutionConfig, evolve_population
from fidp.search.features import FeatureConfig
from fidp.search.surrogate import GraphSurrogateModel, SurrogateConfig
from fidp.search.train_surrogate import TrainingConfig, TrainingResult, train_surrogate
from fidp.search.utils import set_seed


class Evaluator(Protocol):
    """Evaluator protocol for circuit designs."""

    def __call__(self, design: DesignRecord) -> np.ndarray:  # pragma: no cover - typing hook
        ...


@dataclass
class ActiveLearningResult:
    """Result of an active learning iteration."""

    new_designs: List[DesignRecord]
    dataset_size: int
    model: Optional[GraphSurrogateModel]
    training_result: Optional[TrainingResult]


class ActiveLearningLoop:
    """Active learning loop for surrogate-driven search."""

    def __init__(
        self,
        evaluator: Evaluator,
        feature_config: FeatureConfig,
        surrogate_config: SurrogateConfig,
        training_config: Optional[TrainingConfig] = None,
        objective_signs: Optional[Sequence[float]] = None,
    ) -> None:
        self._evaluator = evaluator
        self._feature_config = feature_config
        self._surrogate_config = surrogate_config
        self._training_config = training_config or TrainingConfig()
        self._objective_signs = objective_signs
        self._cache: Dict[str, np.ndarray] = {}
        self._samples: List[CircuitSample] = []
        self._designs: List[DesignRecord] = []
        self._design_keys: set[str] = set()
        self._model: Optional[GraphSurrogateModel] = None

    def dataset(self) -> List[CircuitSample]:
        return list(self._samples)

    def evaluate(self, design: DesignRecord) -> np.ndarray:
        """Evaluate a design with caching."""
        key = self._design_key(design)
        if key in self._cache:
            return self._cache[key]
        metrics = np.asarray(self._evaluator(design), dtype=np.float64)
        self._cache[key] = metrics
        return metrics

    def add_designs(self, designs: Iterable[DesignRecord]) -> None:
        """Evaluate and store designs in the dataset."""
        for design in designs:
            key = self._design_key(design)
            if key in self._design_keys:
                continue
            metrics = self.evaluate(design)
            self._samples.append(
                CircuitSample(
                    graph=design.graph,
                    targets=metrics,
                    global_features=design.global_features,
                )
            )
            self._designs.append(design)
            self._design_keys.add(key)

    def train(self) -> TrainingResult:
        """Train the surrogate model on the current dataset."""
        result = train_surrogate(
            samples=self._samples,
            surrogate_config=self._surrogate_config,
            feature_config=self._feature_config,
            training_config=self._training_config,
        )
        self._model = result.model
        return result

    def run_iteration(
        self,
        initial_designs: Sequence[DesignRecord],
        bo_bounds: Optional[Sequence[Sequence[float]]] = None,
        batch_size: int = 4,
        seed: Optional[int] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        design_factory: Optional[Callable[[np.ndarray], DesignRecord]] = None,
    ) -> ActiveLearningResult:
        """Run a single active learning iteration."""
        set_seed(seed)
        self.add_designs(initial_designs)
        training_result = self.train()

        new_designs: List[DesignRecord] = []

        if bo_bounds is not None and batch_size > 0:
            X, Y = self._collect_observations()
            if is_botorch_available():
                candidates = propose_next_botorch(
                    bounds=bo_bounds,
                    X=X,
                    Y=Y,
                    batch_size=batch_size,
                    seed=seed,
                    objective_signs=self._objective_signs,
                )
            else:
                candidates = propose_next_random(
                    bounds=bo_bounds,
                    batch_size=batch_size,
                    seed=seed,
                )
            new_designs.extend(self._build_designs_from_candidates(candidates, initial_designs, design_factory))

        if evolution_config is not None:
            evo_result = evolve_population(
                population=initial_designs,
                evaluate_fn=self.evaluate,
                config=evolution_config,
                seed=seed,
            )
            new_designs.extend([member.design for member in evo_result.population])

        # Deduplicate and limit to batch_size.
        deduped: List[DesignRecord] = []
        for design in new_designs:
            if len(deduped) >= batch_size:
                break
            key = self._design_key(design)
            if key in self._design_keys:
                continue
            deduped.append(design)

        self.add_designs(deduped)

        return ActiveLearningResult(
            new_designs=deduped,
            dataset_size=len(self._samples),
            model=self._model,
            training_result=training_result,
        )

    def _collect_observations(self) -> tuple[np.ndarray, np.ndarray]:
        params = []
        targets = []
        for design in self._designs:
            params.append(np.asarray(design.parameters, dtype=np.float64))
            targets.append(np.asarray(self._cache[self._design_key(design)], dtype=np.float64))
        if not params:
            raise ValueError("No parameters available for BO proposals.")
        return np.stack(params, axis=0), np.stack(targets, axis=0)

    def _build_designs_from_candidates(
        self,
        candidates: np.ndarray,
        initial_designs: Sequence[DesignRecord],
        design_factory: Optional[Callable[[np.ndarray], DesignRecord]],
    ) -> List[DesignRecord]:
        if design_factory is not None:
            return [design_factory(params) for params in candidates]
        if not initial_designs:
            raise ValueError("initial_designs are required when design_factory is not provided.")
        template = initial_designs[0]
        return [
            DesignRecord(
                graph=template.graph,
                parameters=np.asarray(params, dtype=np.float64),
                spec=template.spec,
                global_features=template.global_features,
                metadata=dict(template.metadata),
            )
            for params in candidates
        ]

    def _design_key(self, design: DesignRecord) -> str:
        payload = {
            "components": [
                {
                    "kind": comp.kind,
                    "node_a": comp.node_a,
                    "node_b": comp.node_b,
                    "value": comp.value,
                    "depth": comp.depth,
                }
                for comp in design.graph.components
            ],
            "ports": [
                {"pos": port.pos, "neg": port.neg} for port in getattr(design.graph, "ports", [])
            ],
            "parameters": np.asarray(design.parameters, dtype=np.float64).tolist(),
            "spec": design.spec,
            "global_features": None
            if design.global_features is None
            else np.asarray(design.global_features, dtype=np.float64).tolist(),
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
