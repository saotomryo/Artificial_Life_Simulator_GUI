from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.config import SimulationConfig
from app.core.controller import SimulationController

NumberWidget = Union[QDoubleSpinBox, QSpinBox]


class ParameterEditor(QWidget):
    """
    Basic data-driven form for editing simulation parameters.
    Designed so new parameter groups can be added without touching UI logic.
    """

    def __init__(self, config: SimulationConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = deepcopy(config)
        self._number_controls: Dict[Tuple[str, str], NumberWidget] = {}
        self._bool_controls: Dict[Tuple[str, str], QCheckBox] = {}
        self._build_ui()

    def value(self) -> SimulationConfig:
        cfg = deepcopy(self._config)
        for (section, field), widget in self._number_controls.items():
            section_obj = getattr(cfg, section)
            setattr(section_obj, field, widget.value())
        for (section, field), widget in self._bool_controls.items():
            section_obj = getattr(cfg, section)
            setattr(section_obj, field, widget.isChecked())
        return cfg

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._add_world_section(layout)
        self._add_resource_section(layout)
        self._add_metabolism_section(layout)
        self._add_brain_section(layout)
        layout.addStretch(1)

    def _add_world_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("World")
        form = QFormLayout(group)
        form.addRow("Width", self._number_box("world", "width", min_val=200.0, max_val=10000.0, step=100.0))
        form.addRow("Height", self._number_box("world", "height", min_val=200.0, max_val=10000.0, step=100.0))
        form.addRow("Time Step", self._number_box("world", "time_step", decimals=3, step=0.01, min_val=0.01, max_val=1.0))
        form.addRow(
            "Initial Population",
            self._number_box("world", "initial_population", widget_type=QSpinBox, min_val=1, max_val=2000, step=1),
        )
        layout.addWidget(group)

    def _add_resource_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Resources")
        form = QFormLayout(group)
        form.addRow(
            "Initial Food Pieces",
            self._number_box("resources", "initial_food_pieces", widget_type=QSpinBox, min_val=0, max_val=10000, step=50),
        )
        chk = QCheckBox()
        chk.setChecked(self._config.resources.initial_food_scale)
        self._bool_controls[("resources", "initial_food_scale")] = chk
        chk.toggled.connect(self._sync_config)
        form.addRow("Scale Food by Area", chk)
        form.addRow(
            "Food Spawn Rate",
            self._number_box("resources", "food_spawn_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "Food Energy",
            self._number_box("resources", "food_energy", decimals=1, step=5.0, min_val=0.0, max_val=500.0),
        )
        form.addRow(
            "Body Decay Rate",
            self._number_box("resources", "decay_body_rate", decimals=3, step=0.001, min_val=0.90, max_val=1.0),
        )
        layout.addWidget(group)

    def _add_metabolism_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Metabolism")
        form = QFormLayout(group)
        form.addRow("Base Cost", self._number_box("metabolism", "base_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0))
        form.addRow("Idle Cost", self._number_box("metabolism", "idle_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0))
        form.addRow(
            "Starvation Threshold",
            self._number_box("metabolism", "starvation_energy", decimals=1, step=10.0, min_val=0.0, max_val=500.0),
        )
        form.addRow(
            "Starvation Cost",
            self._number_box("metabolism", "starvation_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0),
        )
        form.addRow(
            "Move Cost Coefficient",
            self._number_box("metabolism", "move_cost_k", decimals=5, step=0.0005, min_val=0.0, max_val=0.02),
        )
        form.addRow(
            "Brain Cost per Connection",
            self._number_box("metabolism", "brain_cost_per_conn", decimals=5, step=0.0001, min_val=0.0, max_val=0.01),
        )
        layout.addWidget(group)

    def _add_brain_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Brain & Mutation")
        form = QFormLayout(group)
        form.addRow(
            "Sensor Rays",
            self._number_box("brain", "rays", widget_type=QSpinBox, min_val=1, max_val=32, step=1),
        )
        form.addRow(
            "Sense Range",
            self._number_box("brain", "sense_range", decimals=1, step=10.0, min_val=10.0, max_val=1000.0),
        )
        form.addRow(
            "Weight Sigma",
            self._number_box("brain", "weight_sigma", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "Add Connection Rate",
            self._number_box("brain", "add_connection_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "Add Node Rate",
            self._number_box("brain", "add_node_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "Delete Connection Rate",
            self._number_box("brain", "delete_connection_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        layout.addWidget(group)

    def _number_box(
        self,
        section: str,
        field: str,
        *,
        widget_type=QDoubleSpinBox,
        min_val: float,
        max_val: float,
        step: float,
        decimals: int | None = None,
    ) -> NumberWidget:
        value = getattr(getattr(self._config, section), field)
        widget: NumberWidget = widget_type()
        widget.setMinimum(min_val)
        widget.setMaximum(max_val)
        widget.setSingleStep(step)
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
            widget.setDecimals(decimals if decimals is not None else 2)
        else:
            widget.setValue(int(value))
        widget.valueChanged.connect(self._sync_config)
        self._number_controls[(section, field)] = widget
        return widget

    def _sync_config(self) -> None:
        # Update backing store for serialization convenience.
        for (section, field), widget in self._number_controls.items():
            section_obj = getattr(self._config, section)
            setattr(section_obj, field, widget.value())
        for (section, field), widget in self._bool_controls.items():
            section_obj = getattr(self._config, section)
            setattr(section_obj, field, widget.isChecked())


class SimulationStatsWidget(QGroupBox):
    """Displays live metrics emitted from the simulation backend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Live Stats", parent)
        self._fields: Dict[str, Tuple[str, Callable[[float], str]]] = {
            "tick": ("Tick", lambda v: f"{int(v)}"),
            "population": ("Population", lambda v: f"{int(v)}"),
            "mean_energy": ("Mean Energy", lambda v: f"{float(v):.1f}"),
            "births": ("Births", lambda v: f"{int(v)}"),
            "deaths": ("Deaths", lambda v: f"{int(v)}"),
            "food": ("Food Pieces", lambda v: f"{int(v)}"),
            "bodies": ("Bodies", lambda v: f"{int(v)}"),
        }
        self._labels: Dict[str, QLabel] = {}
        layout = QFormLayout(self)
        for key, (title, _) in self._fields.items():
            lbl = QLabel("0")
            lbl.setObjectName(f"stat_{key}")
            layout.addRow(title, lbl)
            self._labels[key] = lbl

    def update_stats(self, stats: Dict) -> None:
        for key, (_, formatter) in self._fields.items():
            value = stats.get(key)
            if value is None:
                continue
            try:
                text = formatter(value)
            except Exception:
                text = str(value)
            self._labels[key].setText(text)


class WorldViewWidget(QWidget):
    """Simple top-down rendering of the simulation world."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: Dict | None = None
        self.setMinimumSize(420, 420)

    def update_frame(self, frame: Dict | None) -> None:
        self._frame = frame
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(18, 20, 26))
        if not self._frame:
            painter.end()
            return

        frame = self._frame
        world_w = max(1.0, float(frame.get("width", 1.0)))
        world_h = max(1.0, float(frame.get("height", 1.0)))

        margin = 20.0
        avail_w = max(1.0, self.width() - 2 * margin)
        avail_h = max(1.0, self.height() - 2 * margin)
        scale = min(avail_w / world_w, avail_h / world_h)
        offset_x = (self.width() - world_w * scale) / 2.0
        offset_y = (self.height() - world_h * scale) / 2.0

        painter.setPen(QPen(QColor(80, 90, 110)))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(offset_x, offset_y, world_w * scale, world_h * scale))

        def to_px(x: float, y: float) -> Tuple[float, float]:
            return offset_x + x * scale, offset_y + y * scale

        # Draw foods
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(70, 200, 120)))
        for food in frame.get("foods", []):
            fx, fy = to_px(float(food.get("x", 0.0)), float(food.get("y", 0.0)))
            size = max(2.0, 4.0 * scale)
            painter.drawRect(QRectF(fx - size / 2.0, fy - size / 2.0, size, size))

        # Draw bodies
        painter.setBrush(QBrush(QColor(170, 110, 60, 180)))
        for body in frame.get("bodies", []):
            bx, by = to_px(float(body.get("x", 0.0)), float(body.get("y", 0.0)))
            radius = max(4.0, 6.0 + 0.02 * body.get("energy", 0.0)) * scale
            painter.drawEllipse(QRectF(bx - radius, by - radius, radius * 2.0, radius * 2.0))

        # Draw agents
        for agent in frame.get("agents", []):
            ax, ay = to_px(float(agent.get("x", 0.0)), float(agent.get("y", 0.0)))
            size = float(agent.get("size", 1.0))
            energy = float(agent.get("energy", 0.0))
            base_radius = 10.0 + 6.0 * (size - 1.0)
            radius = max(8.0, base_radius * scale)
            energy_norm = max(0.0, min(1.0, energy / 300.0))
            color = QColor.fromHslF(0.32 - 0.32 * energy_norm, 0.6, 0.5)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(20, 20, 20, 160)))
            painter.drawEllipse(QRectF(ax - radius, ay - radius, radius * 2.0, radius * 2.0))

        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, controller: SimulationController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Artificial Life Simulator")
        self.resize(1200, 800)
        self._build_menu()
        self._build_ui()
        self._connect_signals()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        sim_menu = self.menuBar().addMenu("&Simulation")
        self._start_action = sim_menu.addAction("&Start")
        self._stop_action = sim_menu.addAction("S&top")
        self._stop_action.setEnabled(False)
        self._start_action.triggered.connect(self.start_simulation)
        self._stop_action.triggered.connect(self.stop_simulation)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)

        self.parameter_editor = ParameterEditor(self.controller.config)
        layout.addWidget(self.parameter_editor, stretch=2)

        right_side = QWidget()
        right_layout = QVBoxLayout(right_side)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.status_label = QLabel("Tick: 0 | Population: 0 | Mean Energy: 0.0")
        self.status_label.setObjectName("statusLabel")
        right_layout.addWidget(self.status_label)
        self.world_view = WorldViewWidget()
        right_layout.addWidget(self.world_view, stretch=3)
        self.stats_widget = SimulationStatsWidget()
        right_layout.addWidget(self.stats_widget, stretch=1)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        right_layout.addLayout(button_row)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        right_layout.addWidget(QLabel("Event Log"))
        right_layout.addWidget(self.log_output, stretch=1)

        layout.addWidget(right_side, stretch=3)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.controller.state_updated.connect(self._on_state_updated)
        self.controller.config_changed.connect(self._on_config_changed)
        self.controller.simulation_started.connect(self._on_simulation_started)
        self.controller.simulation_stopped.connect(self._on_simulation_stopped)
        self.controller.log_emitted.connect(self._append_log)

    def start_simulation(self) -> None:
        cfg = self.parameter_editor.value()
        self.controller.update_config(cfg)
        try:
            self.controller.start()
        except Exception as exc:  # pragma: no cover - UI feedback
            QMessageBox.critical(self, "Simulation Error", str(exc))

    def stop_simulation(self) -> None:
        self.controller.stop()

    def _on_state_updated(self, payload: Dict) -> None:
        frame = None
        if "state" in payload:
            stats = dict(payload["state"])
            resources = payload.get("resources", {})
            stats["food"] = resources.get("food", stats.get("food", 0))
            stats["bodies"] = resources.get("bodies", stats.get("bodies", 0))
            frame = payload.get("frame")
        else:
            stats = dict(payload)
            frame = stats.pop("frame", None)

        tick = int(stats.get("tick", 0))
        population = int(stats.get("population", 0))
        mean_energy = float(stats.get("mean_energy", 0.0))
        self.status_label.setText(f"Tick: {tick} | Population: {population} | Mean Energy: {mean_energy:.1f}")
        self.stats_widget.update_stats(stats)
        if frame:
            self.world_view.update_frame(frame)

    def _on_config_changed(self, config_dict: Dict) -> None:
        self._append_log("Configuration updated.")

    def _on_simulation_started(self) -> None:
        self.statusBar().showMessage("Simulation running")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._start_action.setEnabled(False)
        self._stop_action.setEnabled(True)

    def _on_simulation_stopped(self) -> None:
        self.statusBar().showMessage("Simulation stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)

    def _append_log(self, message: str) -> None:
        self.log_output.append(message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.controller:
            self.controller.stop()
        super().closeEvent(event)
