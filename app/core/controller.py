from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from .config import SimulationConfig, DEFAULT_CONFIG
from .simulation_backend import SimulationBackend, SimulationState


class SimulationWorker(QThread):
    progressed = Signal(object)
    stopped = Signal()

    def __init__(self, backend: SimulationBackend, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._stop_flag = threading.Event()

    def run(self) -> None:
        self._stop_flag.clear()
        while not self._stop_flag.is_set():
            state = self._backend.step()
            self.progressed.emit(state)
        self.stopped.emit()

    def request_stop(self) -> None:
        self._stop_flag.set()


class SimulationController(QObject):
    config_changed = Signal(dict)
    state_updated = Signal(dict)
    simulation_started = Signal()
    simulation_stopped = Signal()
    log_emitted = Signal(str)

    def __init__(
        self,
        backend: SimulationBackend,
        config: Optional[SimulationConfig] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or DEFAULT_CONFIG
        self._backend = backend
        self._backend.configure(self._config)
        self._worker = SimulationWorker(self._backend)
        self._worker.progressed.connect(self._on_progress)
        self._worker.stopped.connect(self._on_worker_stopped)

    @property
    def config(self) -> SimulationConfig:
        return self._config

    def update_config(self, config: SimulationConfig) -> None:
        self._config = config
        self._backend.configure(config)
        self.config_changed.emit(config.to_dict())

    def start(self) -> None:
        if self._worker.isRunning():
            return
        self._backend.start()
        self._worker.start()
        self.simulation_started.emit()
        self.log_emitted.emit("Simulation started.")

    def stop(self) -> None:
        if not self._worker.isRunning():
            return
        self._worker.request_stop()
        self._worker.wait()
        self._backend.stop()

    def request_snapshot(self) -> None:
        snapshot = self._backend.snapshot()
        self.state_updated.emit(snapshot)

    def _on_progress(self, state: SimulationState) -> None:
        self.state_updated.emit(
            {
                "tick": state.tick,
                "population": state.population,
                "mean_energy": state.mean_energy,
                "births": state.births,
                "deaths": state.deaths,
                "food": state.food_count,
                "bodies": state.body_count,
                "frame": state.frame,
            }
        )

    def _on_worker_stopped(self) -> None:
        self.simulation_stopped.emit()
        self.log_emitted.emit("Simulation stopped.")
