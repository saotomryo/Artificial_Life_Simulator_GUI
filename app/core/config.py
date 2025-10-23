from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Tuple


@dataclass
class WorldConfig:
    width: float = 2000.0
    height: float = 2000.0
    time_step: float = 0.15
    initial_population: int = 50

    def to_tuple(self) -> Tuple[float, float]:
        return self.width, self.height


@dataclass
class ResourceConfig:
    initial_food_pieces: int = 2000
    initial_food_scale: bool = True
    food_spawn_rate: float = 0.05
    food_energy: float = 50.0
    decay_body_rate: float = 0.995
    food_density_variation: float = 0.0


@dataclass
class MetabolismConfig:
    base_cost: float = 0.35
    idle_cost: float = 0.45
    starvation_energy: float = 120.0
    starvation_cost: float = 0.55
    move_cost_k: float = 0.001
    brain_cost_per_conn: float = 0.0006


@dataclass
class BrainConfig:
    rays: int = 5
    sense_range: float = 180.0
    weight_sigma: float = 0.08
    add_connection_rate: float = 0.20
    add_node_rate: float = 0.08
    delete_connection_rate: float = 0.02


@dataclass
class SimulationConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_from_mapping(self, data: Dict[str, Any]) -> None:
        """Merge settings from a nested mapping into the config."""
        for section_name, section_values in data.items():
            section = getattr(self, section_name, None)
            if section is None:
                continue
            if not isinstance(section_values, dict):
                continue
            for key, value in section_values.items():
                if hasattr(section, key):
                    setattr(section, key, value)

    def iter_sections(self) -> Iterable[Tuple[str, Any]]:
        yield "world", self.world
        yield "resources", self.resources
        yield "metabolism", self.metabolism
        yield "brain", self.brain


DEFAULT_CONFIG = SimulationConfig()
