from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import QApplication

from app.core import (
    SimulationConfig,
    SimulationController,
    StubSimulationBackend,
    NeatSimulationBackend,
)
from app.ui import MainWindow


def create_application(argv: Sequence[str]) -> QApplication:
    app = QApplication(list(argv))
    app.setApplicationName("Artificial Life Simulator")
    return app


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Artificial Life Simulator GUI")
    parser.add_argument(
        "--backend",
        choices=("neat", "stub"),
        default="neat",
        help="Simulation backend to use (default: neat)",
    )
    parser.add_argument(
        "--sim-module",
        default="evo_sim_neat_diverse",
        help="Importable module name for the simulation logic (default: evo_sim_neat_diverse)",
    )
    parser.add_argument(
        "--sim-file",
        type=str,
        help="Path to a simulation module (.py) to load when the module name is not importable",
    )
    parser.add_argument(
        "--tick-substeps",
        type=int,
        default=1,
        help="Number of simulation substeps to advance per GUI tick",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.01,
        help="Seconds to sleep between backend step calls",
    )
    return parser.parse_known_args(argv)


def build_backend(args: argparse.Namespace):
    if args.backend == "stub":
        return StubSimulationBackend()
    module_path = Path(args.sim_file).expanduser().resolve() if args.sim_file else None
    return NeatSimulationBackend(
        module_name=args.sim_module,
        module_path=module_path,
        substeps=args.tick_substeps,
        sleep_interval=args.sleep_interval,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    args, qt_args = parse_args(argv[1:])
    qt_argv = [argv[0], *qt_args]

    app = create_application(qt_argv)
    backend = build_backend(args)
    controller = SimulationController(backend, SimulationConfig())
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
