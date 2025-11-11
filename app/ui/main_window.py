from __future__ import annotations

import json
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from PySide6.QtCore import Qt, QRectF, Signal, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dataclasses import dataclass, asdict

from app.core.config import SimulationConfig
from app.core.controller import SimulationController
from app.state_manager import StateManager

NumberWidget = Union[QDoubleSpinBox, QSpinBox]


class SignalBlocker:
    """Context manager to temporarily suppress widget signals."""

    def __init__(self, widget) -> None:
        self._widget = widget
        self._previous = False

    def __enter__(self):
        self._previous = self._widget.blockSignals(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._widget.blockSignals(self._previous)
        return False


class ParameterEditor(QWidget):
    """
    Basic data-driven form for editing simulation parameters.
    Designed so new parameter groups can be added without touching UI logic.
    """

    section_save_requested = Signal(str)
    section_load_requested = Signal(str)

    def __init__(self, config: SimulationConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = deepcopy(config)
        self._number_controls: Dict[Tuple[str, str], NumberWidget] = {}
        self._bool_controls: Dict[Tuple[str, str], QCheckBox] = {}
        self._sections = ["world", "resources", "metabolism", "brain", "environment"]
        self._section_labels = {
            "world": "ワールド",
            "resources": "資源",
            "metabolism": "代謝",
            "brain": "脳と変異",
            "environment": "環境",
        }
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
        self._add_environment_section(layout)
        layout.addStretch(1)
    def sections(self) -> Tuple[str, ...]:
        return tuple(self._sections)

    def section_label(self, section: str) -> str:
        return self._section_labels.get(section, section)

    def _add_world_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox(self._section_labels["world"])
        vbox = QVBoxLayout(group)
        form = QFormLayout()
        form.addRow("幅", self._number_box("world", "width", min_val=200.0, max_val=10000.0, step=100.0))
        form.addRow("高さ", self._number_box("world", "height", min_val=200.0, max_val=10000.0, step=100.0))
        form.addRow("時間刻み", self._number_box("world", "time_step", decimals=3, step=0.01, min_val=0.01, max_val=1.0))
        form.addRow(
            "初期個体数",
            self._number_box("world", "initial_population", widget_type=QSpinBox, min_val=1, max_val=2000, step=1),
        )
        vbox.addLayout(form)
        vbox.addLayout(self._section_buttons_layout("world"))
        layout.addWidget(group)

    def _add_resource_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox(self._section_labels["resources"])
        vbox = QVBoxLayout(group)
        form = QFormLayout()
        form.addRow(
            "初期フード数",
            self._number_box("resources", "initial_food_pieces", widget_type=QSpinBox, min_val=0, max_val=10000, step=50),
        )
        chk = QCheckBox()
        chk.setChecked(self._config.resources.initial_food_scale)
        self._bool_controls[("resources", "initial_food_scale")] = chk
        chk.toggled.connect(self._sync_config)
        form.addRow("面積でフードをスケーリング", chk)
        form.addRow(
            "フード出現率",
            self._number_box("resources", "food_spawn_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "フードエネルギー",
            self._number_box("resources", "food_energy", decimals=1, step=5.0, min_val=0.0, max_val=500.0),
        )
        form.addRow(
            "死体減衰率",
            self._number_box("resources", "decay_body_rate", decimals=3, step=0.001, min_val=0.90, max_val=1.0),
        )
        form.addRow(
            "密度変動σ",
            self._number_box("resources", "food_density_variation", decimals=2, step=0.05, min_val=0.0, max_val=2.0),
        )
        vbox.addLayout(form)
        vbox.addLayout(self._section_buttons_layout("resources"))
        layout.addWidget(group)

    def _add_metabolism_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox(self._section_labels["metabolism"])
        vbox = QVBoxLayout(group)
        form = QFormLayout()
        form.addRow("基礎消費", self._number_box("metabolism", "base_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0))
        form.addRow("アイドル消費", self._number_box("metabolism", "idle_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0))
        form.addRow(
            "飢餓エネルギー閾値",
            self._number_box("metabolism", "starvation_energy", decimals=1, step=10.0, min_val=0.0, max_val=500.0),
        )
        form.addRow(
            "飢餓追加消費",
            self._number_box("metabolism", "starvation_cost", decimals=3, step=0.01, min_val=0.0, max_val=5.0),
        )
        form.addRow(
            "移動コスト係数",
            self._number_box("metabolism", "move_cost_k", decimals=5, step=0.0005, min_val=0.0, max_val=0.02),
        )
        form.addRow(
            "結合あたり脳コスト",
            self._number_box("metabolism", "brain_cost_per_conn", decimals=5, step=0.0001, min_val=0.0, max_val=0.01),
        )
        form.addRow(
            "分裂バイアス",
            self._number_box("metabolism", "fission_bias", decimals=2, step=0.1, min_val=0.1, max_val=3.0),
        )
        vbox.addLayout(form)
        vbox.addLayout(self._section_buttons_layout("metabolism"))
        layout.addWidget(group)

    def _add_brain_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox(self._section_labels["brain"])
        vbox = QVBoxLayout(group)
        form = QFormLayout()
        form.addRow(
            "センサーレイ数",
            self._number_box("brain", "rays", widget_type=QSpinBox, min_val=1, max_val=32, step=1),
        )
        form.addRow(
            "感知距離",
            self._number_box("brain", "sense_range", decimals=1, step=10.0, min_val=10.0, max_val=1000.0),
        )
        form.addRow(
            "重み摂動σ",
            self._number_box("brain", "weight_sigma", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "結合追加率",
            self._number_box("brain", "add_connection_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "ノード追加率",
            self._number_box("brain", "add_node_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "結合削除率",
            self._number_box("brain", "delete_connection_rate", decimals=3, step=0.01, min_val=0.0, max_val=1.0),
        )
        vbox.addLayout(form)
        vbox.addLayout(self._section_buttons_layout("brain"))
        layout.addWidget(group)

    def _add_environment_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox(self._section_labels["environment"])
        vbox = QVBoxLayout(group)
        form = QFormLayout()
        form.addRow(
            "季節周期",
            self._number_box("environment", "season_period", decimals=0, step=100.0, min_val=100.0, max_val=20000.0),
        )
        form.addRow(
            "季節振幅",
            self._number_box("environment", "season_amplitude", decimals=2, step=0.05, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "危険強度",
            self._number_box("environment", "hazard_strength", decimals=2, step=0.05, min_val=0.0, max_val=1.0),
        )
        form.addRow(
            "危険領域割合",
            self._number_box("environment", "hazard_coverage", decimals=2, step=0.05, min_val=0.0, max_val=1.0),
        )
        vbox.addLayout(form)
        vbox.addLayout(self._section_buttons_layout("environment"))
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

    def export_section(self, section: str) -> Dict:
        section_obj = getattr(self._config, section)
        return asdict(section_obj)

    def import_section(self, section: str, data: Dict) -> None:
        section_obj = getattr(self._config, section)
        for key, value in data.items():
            if hasattr(section_obj, key):
                current = getattr(section_obj, key)
                if isinstance(current, bool):
                    coerced = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    coerced = int(value)
                elif isinstance(current, float):
                    coerced = float(value)
                else:
                    coerced = value
                setattr(section_obj, key, coerced)
                self._set_widget_value(section, key, coerced)
        self._sync_config()

    def _set_widget_value(self, section: str, field: str, value) -> None:
        widget = self._number_controls.get((section, field))
        if widget is not None:
            with SignalBlocker(widget):
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                else:
                    widget.setValue(int(value))
            return
        chk = self._bool_controls.get((section, field))
        if chk is not None:
            with SignalBlocker(chk):
                chk.setChecked(bool(value))

    def _section_buttons_layout(self, section: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addStretch(1)
        save_btn = QPushButton("保存…")
        load_btn = QPushButton("読み込み…")
        save_btn.clicked.connect(lambda: self.section_save_requested.emit(section))
        load_btn.clicked.connect(lambda: self.section_load_requested.emit(section))
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        return layout


class SimulationStatsWidget(QGroupBox):
    """Displays live metrics emitted from the simulation backend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("ライブ統計", parent)
        self._fields: Dict[str, Tuple[str, Callable[[float], str]]] = {
            "tick": ("経過Tick", lambda v: f"{int(v)}"),
            "population": ("個体数", lambda v: f"{int(v)}"),
            "mean_energy": ("平均エネルギー", lambda v: f"{float(v):.1f}"),
            "births": ("出生数", lambda v: f"{int(v)}"),
            "deaths": ("死亡数", lambda v: f"{int(v)}"),
            "food": ("フード数", lambda v: f"{int(v)}"),
            "bodies": ("死体数", lambda v: f"{int(v)}"),
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
        self._food_colors = [
            QColor(70, 200, 120),
            QColor(110, 180, 240),
            QColor(240, 200, 90),
            QColor(230, 140, 200),
            QColor(170, 140, 255),
        ]

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
        for food in frame.get("foods", []):
            fx, fy = to_px(float(food.get("x", 0.0)), float(food.get("y", 0.0)))
            food_type = int(food.get("type", 0))
            energy = float(food.get("energy", 50.0))
            color = self._food_colors[food_type % len(self._food_colors)]
            painter.setBrush(QBrush(color))
            size = max(2.0, 3.0 * scale + 0.015 * energy * scale)
            painter.drawEllipse(QRectF(fx - size / 2.0, fy - size / 2.0, size, size))

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


@dataclass
class RecordedFrame:
    stats: Dict
    frame: Dict | None


class ActionRecorder:
    """Keeps a sequence of state snapshots for later playback."""

    def __init__(self) -> None:
        self._recording = False
        self._frames: list[RecordedFrame] = []

    def start(self) -> None:
        self._frames.clear()
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    def record(self, stats: Dict, frame: Dict | None) -> None:
        if not self._recording:
            return
        self._frames.append(RecordedFrame(stats=deepcopy(stats), frame=deepcopy(frame) if frame else None))

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def has_frames(self) -> bool:
        return bool(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> Sequence[RecordedFrame]:
        return self._frames

    def export_payload(self) -> Dict:
        return {
            "version": 1,
            "frame_count": len(self._frames),
            "frames": [{"stats": frame.stats, "frame": frame.frame} for frame in self._frames],
        }


@dataclass
class StatsSample:
    tick: int
    population: int
    mean_energy: float
    births: int
    deaths: int
    food: int
    bodies: int
    agents_in_frame: int
    avg_agent_energy: float
    avg_agent_size: float
    foods_in_frame: int
    avg_food_energy: float


class StatsRecorder:
    """Collects lightweight metrics for offline analysis."""

    def __init__(self) -> None:
        self._samples: list[StatsSample] = []

    def record(self, stats: Dict, frame: Dict | None) -> None:
        agents = frame.get("agents", []) if frame else []
        foods = frame.get("foods", []) if frame else []
        bodies = frame.get("bodies", []) if frame else []
        avg_agent_energy = (
            sum(float(agent.get("energy", 0.0)) for agent in agents) / len(agents) if agents else 0.0
        )
        avg_agent_size = (
            sum(float(agent.get("size", 0.0)) for agent in agents) / len(agents) if agents else 0.0
        )
        avg_food_energy = (
            sum(float(food.get("energy", 0.0)) for food in foods) / len(foods) if foods else 0.0
        )
        sample = StatsSample(
            tick=int(stats.get("tick", 0)),
            population=int(stats.get("population", 0)),
            mean_energy=float(stats.get("mean_energy", 0.0)),
            births=int(stats.get("births", 0)),
            deaths=int(stats.get("deaths", 0)),
            food=int(stats.get("food", 0)),
            bodies=int(stats.get("bodies", 0)),
            agents_in_frame=len(agents),
            avg_agent_energy=avg_agent_energy,
            avg_agent_size=avg_agent_size,
            foods_in_frame=len(foods),
            avg_food_energy=avg_food_energy,
        )
        self._samples.append(sample)

    def clear(self) -> None:
        self._samples.clear()

    def has_data(self) -> bool:
        return bool(self._samples)

    def sample_count(self) -> int:
        return len(self._samples)

    def export_csv(self, path: Path) -> None:
        headers = [
            "tick",
            "population",
            "mean_energy",
            "births",
            "deaths",
            "food_resources",
            "bodies",
            "agents_in_frame",
            "avg_agent_energy",
            "avg_agent_size",
            "foods_in_frame",
            "avg_food_energy",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for sample in self._samples:
                writer.writerow(
                    [
                        sample.tick,
                        sample.population,
                        f"{sample.mean_energy:.4f}",
                        sample.births,
                        sample.deaths,
                        sample.food,
                        sample.bodies,
                        sample.agents_in_frame,
                        f"{sample.avg_agent_energy:.4f}",
                        f"{sample.avg_agent_size:.4f}",
                        sample.foods_in_frame,
                        f"{sample.avg_food_energy:.4f}",
                    ]
                )


class MainWindow(QMainWindow):
    def __init__(self, controller: SimulationController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.state_manager = StateManager()
        self._file_filter = "JSON ファイル (*.json);;すべてのファイル (*)"
        self.setWindowTitle("人工生命シミュレーター")
        self.resize(1200, 800)
        self._recorder = ActionRecorder()
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(75)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._playback_index = 0
        self._playback_active = False
        self._stats_recorder = StatsRecorder()

        self._build_menu()
        self._build_ui()
        self._connect_signals()
        self._restore_config_files()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        exit_action = file_menu.addAction("終了(&X)")
        exit_action.triggered.connect(self.close)

        sim_menu = self.menuBar().addMenu("シミュレーション(&S)")
        self._start_action = sim_menu.addAction("開始(&A)")
        self._stop_action = sim_menu.addAction("停止(&T)")
        self._stop_action.setEnabled(False)
        self._start_action.triggered.connect(self.start_simulation)
        self._stop_action.triggered.connect(self.stop_simulation)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)

        self.parameter_editor = ParameterEditor(self.controller.config)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.parameter_editor)
        layout.addWidget(left_scroll, stretch=2)

        right_side = QWidget()
        right_layout = QVBoxLayout(right_side)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.status_label = QLabel("経過Tick: 0 | 個体数: 0 | 平均エネルギー: 0.0")
        self.status_label.setObjectName("statusLabel")
        right_layout.addWidget(self.status_label)
        self.world_view = WorldViewWidget()
        right_layout.addWidget(self.world_view, stretch=3)
        self.stats_widget = SimulationStatsWidget()
        right_layout.addWidget(self.stats_widget, stretch=1)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("開始")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("個体を保存")
        self.load_button = QPushButton("個体を読み込み")
        self.record_button = QPushButton("記録開始")
        self.record_button.setCheckable(True)
        self.playback_button = QPushButton("再生")
        self.playback_button.setCheckable(True)
        self.playback_button.setEnabled(False)
        self.export_record_button = QPushButton("記録を保存")
        self.export_record_button.setEnabled(False)
        self.export_stats_button = QPushButton("統計を保存")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.playback_button)
        button_row.addWidget(self.export_record_button)
        button_row.addWidget(self.export_stats_button)
        right_layout.addLayout(button_row)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        right_layout.addWidget(QLabel("イベントログ"))
        right_layout.addWidget(self.log_output, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_side)
        layout.addWidget(right_scroll, stretch=3)
        self.setCentralWidget(central)

        self.statusBar().showMessage("待機中")

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.save_button.clicked.connect(self.save_agents_snapshot)
        self.load_button.clicked.connect(self.load_agents_snapshot)
        self.record_button.toggled.connect(self._toggle_recording)
        self.playback_button.toggled.connect(self._toggle_playback)
        self.export_record_button.clicked.connect(self._export_recording)
        self.export_stats_button.clicked.connect(self._export_stats)
        self.parameter_editor.section_save_requested.connect(self._on_section_save_requested)
        self.parameter_editor.section_load_requested.connect(self._on_section_load_requested)
        self.controller.state_updated.connect(self._on_state_updated)
        self.controller.config_changed.connect(self._on_config_changed)
        self.controller.simulation_started.connect(self._on_simulation_started)
        self.controller.simulation_stopped.connect(self._on_simulation_stopped)
        self.controller.log_emitted.connect(self._append_log)

    def start_simulation(self) -> None:
        cfg = self.parameter_editor.value()
        self.controller.update_config(cfg)
        self._stats_recorder.clear()
        try:
            self.controller.start()
        except Exception as exc:  # pragma: no cover - UI feedback
            QMessageBox.critical(self, "シミュレーションエラー", str(exc))

    def stop_simulation(self) -> None:
        self.controller.stop()

    def save_agents_snapshot(self) -> None:
        suggested = self.state_manager.get_agent_file() or (Path.cwd() / "last10_genomes.json")
        path = self._get_save_path("個体を保存", suggested)
        if path is None:
            return
        try:
            path = path.resolve()
            backup = self._backup_existing_file(path)
            data_path = Path(self.controller.save_agents()).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            if data_path == path:
                pass
            else:
                copy2(data_path, path)
                if data_path.exists():
                    data_path.unlink()
        except Exception as exc:  # pragma: no cover - UI feedback
            QMessageBox.critical(self, "保存に失敗しました", str(exc))
            return
        if backup is not None:
            self._append_log(f"{backup.name} に過去の個体データを退避しました")
        self._append_log(f"個体スナップショットを {path} に保存しました")
        self.statusBar().showMessage(f"個体スナップショットを {path} に保存しました", 5000)
        self.state_manager.set_agent_file(path)
        self.controller.request_snapshot()

    def load_agents_snapshot(self) -> None:
        suggested = self.state_manager.get_agent_file()
        path = self._get_open_path("個体を読み込み", suggested)
        if path is None:
            return
        try:
            self.controller.load_agents(path)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みに失敗しました", str(exc))
            return
        self._append_log(f"{path} から個体を読み込みました")
        self.statusBar().showMessage(f"{path} から個体を読み込みました", 5000)
        self.state_manager.set_agent_file(path)
        self.controller.request_snapshot()

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

        self._render_state(stats, frame)
        self._recorder.record(stats, frame)
        self._stats_recorder.record(stats, frame)

    def _on_config_changed(self, config_dict: Dict) -> None:
        self._append_log("設定を更新しました。")

    def _on_simulation_started(self) -> None:
        self.statusBar().showMessage("シミュレーション実行中")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._start_action.setEnabled(False)
        self._stop_action.setEnabled(True)

    def _on_simulation_stopped(self) -> None:
        self.statusBar().showMessage("シミュレーション停止")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)

    def _append_log(self, message: str) -> None:
        self.log_output.append(message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.controller:
            self.controller.stop()
        self._stop_playback()
        super().closeEvent(event)

    def _backup_existing_file(self, path: Path) -> Optional[Path]:
        if not path.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = path.suffix or ".json"
        stem = path.stem or "snapshot"
        backup = path.with_name(f"{stem}_{timestamp}{suffix}")
        counter = 1
        while backup.exists():
            backup = path.with_name(f"{stem}_{timestamp}_{counter}{suffix}")
            counter += 1
        path.rename(backup)
        return backup

    def _render_state(self, stats: Dict, frame: Dict | None) -> None:
        tick = int(stats.get("tick", 0))
        population = int(stats.get("population", 0))
        mean_energy = float(stats.get("mean_energy", 0.0))
        self.status_label.setText(f"経過Tick: {tick} | 個体数: {population} | 平均エネルギー: {mean_energy:.1f}")
        self.stats_widget.update_stats(stats)
        if frame:
            self.world_view.update_frame(frame)

    def _toggle_recording(self, checked: bool) -> None:
        if checked:
            if self._playback_active:
                self._stop_playback()
            self._recorder.start()
            self.record_button.setText("記録停止")
            self.playback_button.setEnabled(False)
            self.export_record_button.setEnabled(False)
            self._append_log("行動の記録を開始しました。")
            self.statusBar().showMessage("行動の記録を開始しました", 3000)
            return

        self._recorder.stop()
        self.record_button.setText("記録開始")
        frames = self._recorder.frame_count
        self._append_log(f"行動の記録を停止しました（{frames} フレーム）")
        self.statusBar().showMessage(f"行動の記録を停止しました（{frames} フレーム）", 5000)
        self._update_recording_controls()

    def _toggle_playback(self, checked: bool) -> None:
        if checked:
            if not self._recorder.has_frames:
                QMessageBox.information(self, "再生できません", "再生可能な記録がありません。")
                self.playback_button.blockSignals(True)
                self.playback_button.setChecked(False)
                self.playback_button.blockSignals(False)
                return
            self.controller.stop()
            self._start_playback()
            return
        self._stop_playback()

    def _start_playback(self) -> None:
        self._playback_index = 0
        self._playback_active = True
        self._playback_timer.start()
        self.playback_button.setText("再生停止")
        self.record_button.setEnabled(False)
        self.export_record_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._start_action.setEnabled(False)
        self._stop_action.setEnabled(False)
        self.statusBar().showMessage("記録を再生しています…")
        self._append_log("記録の再生を開始しました。")

    def _stop_playback(self) -> None:
        if not self._playback_active:
            self.playback_button.blockSignals(True)
            self.playback_button.setChecked(False)
            self.playback_button.blockSignals(False)
            self.playback_button.setText("再生")
            self.record_button.setEnabled(True)
            self._update_recording_controls()
            return
        self._playback_timer.stop()
        self._playback_active = False
        self._playback_index = 0
        self.playback_button.blockSignals(True)
        self.playback_button.setChecked(False)
        self.playback_button.blockSignals(False)
        self.playback_button.setText("再生")
        self.record_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)
        self.statusBar().showMessage("記録の再生を停止しました", 3000)
        self._append_log("記録の再生を停止しました。")
        self._update_recording_controls()

    def _advance_playback(self) -> None:
        frames = self._recorder.frames
        if self._playback_index >= len(frames):
            self._stop_playback()
            return
        frame = frames[self._playback_index]
        self._render_state(frame.stats, frame.frame)
        self._playback_index += 1
        if self._playback_index >= len(frames):
            self._stop_playback()

    def _update_recording_controls(self) -> None:
        enabled = self._recorder.has_frames and not self._recorder.is_recording and not self._playback_active
        self.playback_button.setEnabled(enabled)
        self.export_record_button.setEnabled(enabled)

    def _export_recording(self) -> None:
        if not self._recorder.has_frames:
            QMessageBox.information(self, "保存できません", "保存可能な記録がありません。")
            return
        suggested = Path.cwd() / "simulation_recording.json"
        path = self._get_save_path("記録を保存", suggested)
        if path is None:
            return
        data = self._recorder.export_payload()
        if self._write_json(path, data):
            frames = self._recorder.frame_count
            self._append_log(f"記録を {path} に保存しました（{frames} フレーム）")
            self.statusBar().showMessage(f"記録を {path} に保存しました", 5000)

    def _export_stats(self) -> None:
        if not self._stats_recorder.has_data():
            QMessageBox.information(self, "保存できません", "保存可能な統計データがありません。")
            return
        suggested = Path.cwd() / "simulation_stats.csv"
        path = self._get_save_path("統計を保存", suggested)
        if path is None:
            return
        try:
            self._stats_recorder.export_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "保存に失敗しました", f"統計データの書き込みに失敗しました:\n{exc}")
            return
        self._append_log(f"統計データを {path} に保存しました（{self._stats_recorder.sample_count()} 行）")
        self.statusBar().showMessage(f"統計データを {path} に保存しました", 5000)

    def _on_section_save_requested(self, section: str) -> None:
        label = self.parameter_editor.section_label(section)
        data = self.parameter_editor.export_section(section)
        suggested = self._suggest_config_path(section, for_save=True)
        path = self._get_save_path(f"{label}設定を保存", suggested)
        if path is None:
            return
        if self._write_json(path, data):
            self.state_manager.set_config_file(section, path)
            self._append_log(f"{label}設定を {path} に保存しました")
            self.statusBar().showMessage(f"{label}設定を {path} に保存しました", 5000)

    def _on_section_load_requested(self, section: str) -> None:
        label = self.parameter_editor.section_label(section)
        suggested = self._suggest_config_path(section, for_save=False)
        path = self._get_open_path(f"{label}設定を読み込み", suggested)
        if path is None:
            return
        if self._load_config_from_path(section, path, notify=True):
            self.state_manager.set_config_file(section, path)
            self.controller.update_config(self.parameter_editor.value())

    def _restore_config_files(self) -> None:
        changed = False
        for section in self.parameter_editor.sections():
            stored = self.state_manager.get_config_file(section)
            if stored is None:
                continue
            if self._load_config_from_path(section, stored, notify=False):
                changed = True
        if changed:
            self.controller.update_config(self.parameter_editor.value())
            self._append_log("前回利用した設定を復元しました。")

    def _suggest_config_path(self, section: str, *, for_save: bool) -> Path:
        stored = self.state_manager.get_config_file(section)
        if stored and stored.exists():
            return stored
        default_name = f"{section}_config.json"
        if stored:
            return stored
        return Path.cwd() / default_name

    def _get_save_path(self, caption: str, suggested: Path) -> Optional[Path]:
        filename, _ = QFileDialog.getSaveFileName(self, caption, str(suggested), self._file_filter)
        if not filename:
            return None
        return Path(filename)

    def _get_open_path(self, caption: str, suggested: Optional[Path]) -> Optional[Path]:
        start = str(suggested) if suggested else str(Path.cwd())
        filename, _ = QFileDialog.getOpenFileName(self, caption, start, self._file_filter)
        if not filename:
            return None
        return Path(filename)

    def _write_json(self, path: Path, data: Dict) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            return True
        except Exception as exc:
            QMessageBox.critical(self, "保存に失敗しました", f"{path} への書き込みに失敗しました:\n{exc}")
            return False

    def _load_config_from_path(self, section: str, path: Path, *, notify: bool) -> bool:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            if notify:
                QMessageBox.critical(self, "読み込みに失敗しました", f"{path} の読み込みに失敗しました:\n{exc}")
            return False
        if not isinstance(data, dict):
            if notify:
                QMessageBox.critical(self, "読み込みに失敗しました", f"{path} の内容が不正です")
            return False
        self.parameter_editor.import_section(section, data)
        label = self.parameter_editor.section_label(section)
        if notify:
            self._append_log(f"{label}設定を {path} から読み込みました")
            self.statusBar().showMessage(f"{label}設定を {path} から読み込みました", 5000)
        return True
