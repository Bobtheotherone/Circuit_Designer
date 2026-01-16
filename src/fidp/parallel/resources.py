"""Resource specifications for Ray tasks and actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResourceSpec:
    """Resource requirements for Ray tasks or actors."""

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_mb: float | None = None
    custom_resources: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.num_cpus < 0:
            raise ValueError("num_cpus must be >= 0.")
        if self.num_gpus < 0:
            raise ValueError("num_gpus must be >= 0.")
        if self.memory_mb is not None and self.memory_mb < 0:
            raise ValueError("memory_mb must be >= 0.")
        if self.custom_resources:
            for name, value in self.custom_resources.items():
                if value < 0:
                    raise ValueError(f"custom resource {name} must be >= 0.")

    def ray_options(self) -> dict[str, Any]:
        """Return Ray options for task or actor creation."""
        options: dict[str, Any] = {
            "num_cpus": float(self.num_cpus),
            "num_gpus": float(self.num_gpus),
        }
        if self.memory_mb is not None:
            options["memory"] = int(self.memory_mb * 1024 * 1024)
        if self.custom_resources:
            options["resources"] = dict(self.custom_resources)
        return options

    def requires_gpu(self) -> bool:
        """Return True if the spec requests GPUs."""
        return self.num_gpus > 0
