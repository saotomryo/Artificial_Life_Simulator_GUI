from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

from app.core.config import DEFAULT_CONFIG, SimulationConfig
from app.core.simulation_backend import SimulationBackend, SimulationState


class GradioSimulationController:
    """UI-agnostic controller tailored for Gradio callbacks."""

    def __init__(
        self,
        backend: SimulationBackend,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self._backend = backend
        self._config = config or DEFAULT_CONFIG
        self._backend.configure(self._config)
        self._lock = threading.Lock()
        self._running = False
        self._last_state: Optional[SimulationState] = None

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    def apply_config(self, config: SimulationConfig) -> None:
        with self._lock:
            self._config = config
            self._backend.configure(config)

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._backend.start()
            self._running = True
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._running = False
            self._backend.stop()
            return True

    def step_or_snapshot(self) -> Dict:
        with self._lock:
            if self._running:
                state = self._backend.step()
                self._last_state = state
                return _state_payload(state)
        snapshot = self._backend.snapshot()
        return _snapshot_payload(snapshot, self._last_state)

    def save_agents(self) -> Path:
        return self._backend.save_agents()

    def load_agents(self, path: Path) -> None:
        self._backend.load_agents(path)


def _state_payload(state: SimulationState) -> Dict:
    payload = {
        "tick": state.tick,
        "population": state.population,
        "mean_energy": state.mean_energy,
        "births": state.births,
        "deaths": state.deaths,
        "food": state.food_count,
        "bodies": state.body_count,
        "frame": state.frame,
        "telemetry": state.telemetry,
    }
    return payload


def _snapshot_payload(snapshot: Dict, last_state: Optional[SimulationState]) -> Dict:
    state = snapshot.get("state", {})
    resources = snapshot.get("resources", {})
    payload = {
        "tick": state.get("tick", getattr(last_state, "tick", 0) if last_state else 0),
        "population": state.get("population", getattr(last_state, "population", 0) if last_state else 0),
        "mean_energy": state.get("mean_energy", getattr(last_state, "mean_energy", 0.0) if last_state else 0.0),
        "births": state.get("births", getattr(last_state, "births", 0) if last_state else 0),
        "deaths": state.get("deaths", getattr(last_state, "deaths", 0) if last_state else 0),
        "food": resources.get("food", getattr(last_state, "food_count", 0) if last_state else 0),
        "bodies": resources.get("bodies", getattr(last_state, "body_count", 0) if last_state else 0),
        "frame": snapshot.get("frame"),
        "telemetry": snapshot.get("telemetry", {}),
    }
    return payload
