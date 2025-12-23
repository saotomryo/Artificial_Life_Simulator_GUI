from __future__ import annotations

import importlib
import importlib.util
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Protocol

from .config import SimulationConfig


@dataclass
class SimulationState:
    tick: int = 0
    population: int = 0
    mean_energy: float = 0.0
    births: int = 0
    deaths: int = 0
    food_count: int = 0
    body_count: int = 0
    frame: Dict | None = None
    telemetry: Dict = field(default_factory=dict)


class SimulationBackend(Protocol):
    """Interface the GUI uses to control a simulation implementation."""

    def configure(self, config: SimulationConfig) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def step(self) -> SimulationState:
        ...

    def snapshot(self) -> Dict:
        ...

    def save_agents(self) -> Path:
        ...

    def load_agents(self, path: Path) -> None:
        ...


class StubSimulationBackend:
    """
    Minimal backend used as a placeholder until the real simulator
    is hooked up. It pretends to advance_ticks and generates dummy stats.
    """

    def __init__(self) -> None:
        self.config = SimulationConfig()
        self._state = SimulationState()
        self._lock = threading.Lock()
        self._running = False
        self._total_births = 0
        self._total_deaths = 0
        self._prev_population = self.config.world.initial_population
        self._phase = 0.0

    def configure(self, config: SimulationConfig) -> None:
        with self._lock:
            self.config = config
            self._state = SimulationState()
            self._total_births = 0
            self._total_deaths = 0
            self._prev_population = config.world.initial_population
            self._phase = 0.0

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def step(self) -> SimulationState:
        time.sleep(0.02)
        with self._lock:
            if self._running:
                prev_population = self._state.population or self._prev_population
                self._state.tick += 1
                # Fake population oscillation for UI verification
                self._state.population = int(
                    self.config.world.initial_population * (1.0 + 0.3 * ((self._state.tick % 40) / 20 - 1))
                )
                self._state.mean_energy = 150.0 + 20.0 * ((self._state.tick % 50) / 25 - 1)
                diff = self._state.population - prev_population
                if diff > 0:
                    self._total_births += diff
                elif diff < 0:
                    self._total_deaths += abs(diff)
                self._state.births = self._total_births
                self._state.deaths = self._total_deaths
                self._state.food_count = max(0, int(150 + 40 * math.sin(self._state.tick / 15.0)))
                self._state.body_count = max(0, int(20 + 10 * math.cos(self._state.tick / 20.0)))
                self._state.frame = self._fake_frame()
                self._state.telemetry = {}
                self._prev_population = self._state.population
        return self._state

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "config": self.config.to_dict(),
                "state": {
                    "tick": self._state.tick,
                    "population": self._state.population,
                    "mean_energy": self._state.mean_energy,
                    "births": self._state.births,
                    "deaths": self._state.deaths,
                },
                "resources": {
                    "food": self._state.food_count,
                    "bodies": self._state.body_count,
                },
                "frame": self._state.frame,
                "telemetry": self._state.telemetry,
            }

    def save_agents(self) -> Path:
        from random import random

        path = Path("last10_genomes.json")
        dummy = []
        for i in range(10):
            dummy.append(
                {
                    "genome": {
                        "nodes": [],
                        "conns": [],
                        "in_ids": [],
                        "out_ids": [],
                    },
                    "S": 1.0 + 0.05 * math.sin(self._state.tick + i),
                    "score": random(),
                }
            )
        path.write_text(json.dumps(dummy, indent=2))
        return path

    def load_agents(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise ValueError(f"Failed to load agents from {path}: {exc}") from exc
        with self._lock:
            self._state.tick = 0
            self._state.population = len(data)
            self._state.mean_energy = 150.0
            self._total_births = self._state.population
            self._total_deaths = 0
            self._state.births = self._total_births
            self._state.deaths = 0
            self._state.frame = {
                "width": self.config.world.width,
                "height": self.config.world.height,
                "agents": [
                    {"x": (i + 1) * 10.0, "y": (i + 1) * 10.0, "size": 1.0, "energy": 200.0}
                    for i in range(self._state.population)
                ],
                "foods": [],
                "bodies": [],
            }
    def _fake_frame(self) -> Dict:
        world_w = self.config.world.width
        world_h = self.config.world.height
        agents = []
        foods = []
        bodies = []
        n_agents = max(5, self._state.population or self.config.world.initial_population)
        for i in range(min(n_agents, 30)):
            angle = (i / max(1, n_agents)) * 2 * math.pi + self._phase
            radius = 0.4 + 0.5 * math.sin(self._state.tick / 50.0 + i)
            agents.append(
                {
                    "x": (0.5 + radius * math.cos(angle)) * world_w,
                    "y": (0.5 + radius * math.sin(angle)) * world_h,
                    "size": 1.0 + 0.2 * math.sin(angle * 2),
                    "energy": 150 + 20 * math.sin(angle + self._state.tick / 20.0),
                }
            )
        for i in range(10):
            foods.append(
                {
                    "x": (0.5 + 0.45 * math.cos(self._phase + i)) * world_w,
                    "y": (0.5 + 0.45 * math.sin(self._phase + i)) * world_h,
                }
            )
        for i in range(5):
            bodies.append(
                {
                    "x": (0.5 + 0.3 * math.cos(-self._phase + i)) * world_w,
                    "y": (0.5 + 0.3 * math.sin(-self._phase + i)) * world_h,
                    "energy": 40 + 10 * math.cos(self._state.tick / 30.0 + i),
                }
            )
        self._phase += 0.03
        return {
            "width": world_w,
            "height": world_h,
            "agents": agents,
            "foods": foods,
            "bodies": bodies,
        }


class NeatSimulationBackend:
    """
    Adapter that drives `evo_sim_neat_diverse.World` through the backend protocol.
    Allows the GUI layer to run the real simulation headlessly.
    """

    def __init__(
        self,
        module_name: str = "app.sim.neat_simulation",
        *,
        module_path: Optional[Path] = None,
        substeps: int = 1,
        sleep_interval: float = 0.01,
    ) -> None:
        self._module_name = module_name
        self._module: ModuleType
        self._module_path: Optional[Path] = None
        self._module = self._load_module(module_name, module_path)
        self._config: SimulationConfig = SimulationConfig()
        self._world: Optional[object] = None
        self._state = SimulationState()
        self._lock = threading.Lock()
        self._running = False
        self._sleep_interval = max(0.0, float(sleep_interval))
        self._substeps = max(1, int(substeps))

    def configure(self, config: SimulationConfig) -> None:
        with self._lock:
            self._config = config
            self._apply_config()
            self._world = self._create_world()
            self._state = SimulationState()

    def start(self) -> None:
        with self._lock:
            self._ensure_world()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def step(self) -> SimulationState:
        time.sleep(self._sleep_interval)
        with self._lock:
            world = self._ensure_world()
            if self._running:
                world.tick(substeps=self._substeps)
            population = len(world.agents)
            mean_energy = (
                sum(agent.E for agent in world.agents) / population if population > 0 else 0.0
            )
            telemetry = {}
            if hasattr(world, "telemetry"):
                try:
                    telemetry = dict(getattr(world, "telemetry"))
                except Exception:
                    telemetry = {}
            self._state.tick = world.t
            self._state.population = population
            self._state.mean_energy = mean_energy
            self._state.births = getattr(world, "births", 0)
            self._state.deaths = getattr(world, "deaths", 0)
            self._state.food_count = len(getattr(world, "foods", []))
            self._state.body_count = len(getattr(world, "bodies", []))
            self._state.frame = self._capture_frame(world)
            self._state.telemetry = telemetry
            return self._state

    def snapshot(self) -> Dict:
        with self._lock:
            world = self._ensure_world()
            return {
                "config": self._config.to_dict(),
                "state": {
                    "tick": world.t,
                    "population": len(world.agents),
                    "mean_energy": self._state.mean_energy,
                    "births": world.births,
                    "deaths": world.deaths,
                },
                "resources": {
                    "food": len(world.foods),
                    "bodies": len(world.bodies),
                },
                "frame": self._capture_frame(world),
                "telemetry": getattr(self._state, "telemetry", {}),
            }

    def save_agents(self) -> Path:
        with self._lock:
            world = self._ensure_world()
            if hasattr(world, "save_last10"):
                world.save_last10()
            last_path = getattr(self._module, "LAST10_PATH", "last10_genomes.json")
            path = Path(last_path)
            if not path.is_absolute() and self._module_path is not None:
                path = (self._module_path.parent / path).resolve()
            return path
    def load_agents(self, path: Path) -> None:
        with self._lock:
            resolved = path.expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            data = json.loads(resolved.read_text())
            world = self._ensure_world()
            # Align the simulator's default save path with the loaded file
            self._module.LAST10_PATH = str(resolved)
            world.agents = []
            if hasattr(world, "_bootstrap_from_last10_diverse"):
                world._bootstrap_from_last10_diverse(data, self._module.N_INIT)
            else:
                # Fallback: recreate agents from dict if available
                if hasattr(world, "Agent") and hasattr(self._module, "Genome"):
                    Genome = getattr(self._module, "Genome")
                    Agent = getattr(self._module, "Agent")
                    for item in data:
                        genome_dict = item.get("genome")
                        genome = Genome.from_dict(genome_dict) if genome_dict else Genome()
                        agent = Agent(genome)
                        if "S" in item:
                            agent.S = item["S"]
                        world.agents.append(agent)
            world.t = 0
            world.births = 0
            world.deaths = 0
            self._state = SimulationState(
                tick=0,
                population=len(world.agents),
                mean_energy=sum(getattr(a, "E", 0.0) for a in world.agents) / len(world.agents)
                if world.agents
                else 0.0,
                births=0,
                deaths=0,
                food_count=len(getattr(world, "foods", [])),
                body_count=len(getattr(world, "bodies", [])),
                frame=self._capture_frame(world),
                telemetry={},
            )

    # ----- internals -----
    def _apply_config(self) -> None:
        sim = self._module
        cfg = self._config

        sim.W = cfg.world.width
        sim.H = cfg.world.height
        sim.DT = cfg.world.time_step
        sim.N_INIT = cfg.world.initial_population

        sim.INITIAL_FOOD_PIECES = cfg.resources.initial_food_pieces
        sim.INITIAL_FOOD_SCALE = cfg.resources.initial_food_scale
        sim.FOOD_RATE = cfg.resources.food_spawn_rate
        sim.FOOD_EN = cfg.resources.food_energy
        sim.DECAY_BODY = cfg.resources.decay_body_rate
        sim.FOOD_DENSITY_VARIATION = cfg.resources.food_density_variation
        sim.FOOD_TYPE_ENERGIES = cfg.resources.food_type_energies
        sim.FOOD_TYPE_WEIGHTS = cfg.resources.food_type_weights

        env = cfg.environment
        sim.SEASON_PERIOD = env.season_period
        sim.SEASON_AMPLITUDE = env.season_amplitude
        sim.HAZARD_STRENGTH = env.hazard_strength
        sim.HAZARD_COVERAGE = env.hazard_coverage

        sim.BASE_COST = cfg.metabolism.base_cost
        sim.IDLE_COST = cfg.metabolism.idle_cost
        sim.STARVATION_E = cfg.metabolism.starvation_energy
        sim.STARVATION_COST = cfg.metabolism.starvation_cost
        sim.MOVE_COST_K = cfg.metabolism.move_cost_k
        sim.BRAIN_COST_PER_CONN = cfg.metabolism.brain_cost_per_conn
        sim.FISSION_RATE_FACTOR = cfg.metabolism.fission_bias
        sim.DASH_VMAX_MULT = cfg.metabolism.dash_vmax_mult
        sim.DASH_COST = cfg.metabolism.dash_cost
        sim.DEFEND_STRENGTH = cfg.metabolism.defend_strength
        sim.DEFEND_COST = cfg.metabolism.defend_cost
        sim.REST_BASE_COST_MULT = cfg.metabolism.rest_base_cost_mult
        sim.REST_COST = cfg.metabolism.rest_cost
 
        sim.N_RAYS = cfg.brain.rays
        sim.R_SENSE = cfg.brain.sense_range
        sim.WEIGHT_SIGMA = cfg.brain.weight_sigma
        sim.P_ADD_CONN = cfg.brain.add_connection_rate
        sim.P_ADD_NODE = cfg.brain.add_node_rate
        sim.P_DEL_CONN = cfg.brain.delete_connection_rate
        sim.ENABLE_ADVANCED_ACTIONS = bool(getattr(cfg.brain, "enable_advanced_actions", False))

        sim.IN_FEATURES = sim.N_RAYS * 3 + 4
        sim.INPUT_IDS = list(range(0, sim.IN_FEATURES))
        sim.OUT_FEATURES = 8 if sim.ENABLE_ADVANCED_ACTIONS else 5
        sim.OUTPUT_IDS = list(range(100, 100 + sim.OUT_FEATURES))

        # Recompute grid to match possible world size changes.
        sim.GRID_W = int(math.ceil(sim.W / sim.CELL))
        sim.GRID_H = int(math.ceil(sim.H / sim.CELL))

        # Reset innovation tracking so each configuration starts fresh.
        sim.INNOV_DB = sim.InnovationDB()
        sim.NEXT_NODE_ID = 1000

    def _create_world(self):
        sim = self._module
        return sim.World(from_last10=False)

    def _ensure_world(self):
        if self._world is None:
            self._world = self._create_world()
        return self._world

    def _capture_frame(self, world) -> Dict:
        sim = self._module
        frame = {
            "width": getattr(sim, "W", 2000.0),
            "height": getattr(sim, "H", 2000.0),
            "agents": [],
            "foods": [],
            "bodies": [],
        }
        hazard_field = getattr(world, "_hazard_field", None)
        if hazard_field is not None:
            try:
                frame["hazard"] = {
                    "grid_w": int(getattr(sim, "GRID_W", 0)),
                    "grid_h": int(getattr(sim, "GRID_H", 0)),
                    "cell": float(getattr(sim, "CELL", 1.0)),
                    "values": hazard_field.astype("float32").ravel().tolist(),
                }
            except Exception:
                pass
        agents = getattr(world, "agents", [])
        for agent in agents:
            frame["agents"].append(
                {
                    "id": int(getattr(agent, "id", -1)),
                    "parent_id": getattr(agent, "parent_id", None),
                    "parent2_id": getattr(agent, "parent2_id", None),
                    "birth_tick": int(getattr(agent, "birth_tick", 0)),
                    "age": int(getattr(agent, "age", 0)),
                    "x": float(agent.x),
                    "y": float(agent.y),
                    "size": float(getattr(agent, "S", 1.0)),
                    "energy": float(getattr(agent, "E", 0.0)),
                    "thrust": float(getattr(agent, "last_thrust", 0.0)),
                    "turn": float(getattr(agent, "last_turn", 0.0)),
                    "attack": bool(getattr(agent, "last_attack", False)),
                    "mate": bool(getattr(agent, "last_mate", False)),
                    "eat_strength": float(getattr(agent, "last_eat_strength", 0.0)),
                    "dash": bool(getattr(agent, "last_dash", False)),
                    "defend": bool(getattr(agent, "last_defend", False)),
                    "rest": bool(getattr(agent, "last_rest", False)),
                    "strategy": str(getattr(agent, "strategy_tag", "")),
                    "food_energy_total": float(getattr(agent, "food_energy_total", 0.0)),
                    "body_energy_total": float(getattr(agent, "body_energy_total", 0.0)),
                    "attack_attempts_total": int(getattr(agent, "attack_attempts_total", 0)),
                    "attack_successes_total": int(getattr(agent, "attack_successes_total", 0)),
                    "mate_attempts_total": int(getattr(agent, "mate_attempts_total", 0)),
                    "mate_successes_total": int(getattr(agent, "mate_successes_total", 0)),
                    "fissions_total": int(getattr(agent, "fissions_total", 0)),
                }
            )
        for food in getattr(world, "foods", []):
            frame["foods"].append(
                {
                    "x": float(food.x),
                    "y": float(food.y),
                    "type": int(getattr(food, "type_id", 0)),
                    "energy": float(getattr(food, "energy", getattr(food, "e", 0.0))),
                }
            )
        for body in getattr(world, "bodies", []):
            frame["bodies"].append({"x": float(body.x), "y": float(body.y), "energy": float(body.e)})
        return frame

    def _load_module(self, module_name: str, module_path: Optional[Path]) -> ModuleType:
        if module_path is not None:
            resolved = module_path.expanduser().resolve()
            if not resolved.exists():
                raise ModuleNotFoundError(f"Simulation module file not found: {resolved}")
            module, _ = self._load_from_file(module_name, resolved)
            self._module_path = resolved
            return module

        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file:
                self._module_path = Path(module_file).resolve()
            return module
        except ModuleNotFoundError:
            candidate = self._find_candidate_file(module_name)
            if candidate is None:
                raise
            module, resolved = self._load_from_file(module_name, candidate)
            self._module_path = resolved
            return module

    def _load_from_file(self, module_name: str, path: Path) -> tuple[ModuleType, Path]:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Unable to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[call-arg]
        return module, path.resolve()

    def _find_candidate_file(self, module_name: str) -> Optional[Path]:
        filename = f"{module_name}.py"
        base = Path(__file__).resolve().parent
        search_dirs = [
            Path.cwd(),
            Path.cwd() / "old",
            base.parent,
            base.parent / "sim",
        ]
        for directory in search_dirs:
            candidate = (directory / filename).resolve()
            if candidate.exists():
                return candidate
        return None
