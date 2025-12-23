from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
except Exception:  # pragma: no cover - runtime environment dependent
    QChart = None  # type: ignore[assignment]
    QChartView = None  # type: ignore[assignment]
    QLineSeries = None  # type: ignore[assignment]
    QValueAxis = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str


_LABELS: Dict[str, str] = {
    "tick": "Tick",
    "population": "個体数",
    "mean_energy": "平均エネルギー",
    "births": "出生数",
    "deaths": "死亡数",
    "food_resources": "フード数",
    "bodies": "死体数",
    "season_mult": "季節係数",
    "hazard_mean": "危険平均",
    "hazard_coverage": "危険カバレッジ",
    "attack_attempts": "攻撃試行",
    "attack_successes": "攻撃成功",
    "mate_attempts": "交配試行",
    "mate_successes": "交配成立",
    "fissions": "分裂回数",
    "deaths_attack": "死亡(攻撃)",
    "deaths_hazard": "死亡(危険)",
    "deaths_energy": "死亡(エネルギー)",
    "mean_speed_frac": "平均速度比",
    "mean_thrust": "平均thrust",
    "mean_turn": "平均turn",
    "mean_eat_strength": "平均摂食強度",
    "eat_energy_food": "摂食エネ(フード)",
    "eat_energy_body": "摂食エネ(死体)",
    "dash_active": "ダッシュ人数",
    "defend_active": "防御人数",
    "rest_active": "休息人数",
    "avg_agent_energy": "平均個体エネ(フレーム)",
    "avg_agent_size": "平均サイズ(フレーム)",
    "foods_in_frame": "フード数(フレーム)",
    "avg_food_energy": "平均フードエネ(フレーム)",
}


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


class StatsViewerWindow(QMainWindow):
    def __init__(self, *, initial_path: Optional[Path] = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("統計ビューア（CSV）")
        self.resize(1100, 750)
        self._path: Optional[Path] = None
        self._rows: list[dict[str, float]] = []
        self._columns: list[str] = []

        self._build_menu()
        self._build_ui()

        if initial_path is not None:
            self.load_csv(initial_path)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        open_action = QAction("CSVを開く…", self)
        open_action.triggered.connect(self._open_dialog)
        file_menu.addAction(open_action)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self._path_label = QLabel("未選択")
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_btn = QPushButton("CSVを開く…")
        open_btn.clicked.connect(self._open_dialog)
        top_row.addWidget(QLabel("ファイル:"))
        top_row.addWidget(self._path_label, stretch=1)
        top_row.addWidget(open_btn)
        layout.addLayout(top_row)

        if QChartView is None:
            msg = QLabel("QtCharts が利用できません（PySide6.QtCharts が読み込めませんでした）。")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self.setCentralWidget(central)
            return

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        self._chart_views: dict[str, QChartView] = {}
        self._chart_lists: dict[str, QListWidget] = {}
        self._chart_series_specs: dict[str, list[SeriesSpec]] = {}
        self._add_chart_tab(
            "人口/資源",
            [
                SeriesSpec("population", _LABELS["population"]),
                SeriesSpec("food_resources", _LABELS["food_resources"]),
                SeriesSpec("bodies", _LABELS["bodies"]),
            ],
        )
        self._add_chart_tab(
            "エネルギー",
            [
                SeriesSpec("mean_energy", _LABELS["mean_energy"]),
                SeriesSpec("avg_agent_energy", _LABELS["avg_agent_energy"]),
                SeriesSpec("eat_energy_food", _LABELS["eat_energy_food"]),
                SeriesSpec("eat_energy_body", _LABELS["eat_energy_body"]),
            ],
        )
        self._add_chart_tab(
            "イベント",
            [
                SeriesSpec("births", _LABELS["births"]),
                SeriesSpec("deaths", _LABELS["deaths"]),
                SeriesSpec("attack_attempts", _LABELS["attack_attempts"]),
                SeriesSpec("attack_successes", _LABELS["attack_successes"]),
                SeriesSpec("mate_attempts", _LABELS["mate_attempts"]),
                SeriesSpec("mate_successes", _LABELS["mate_successes"]),
                SeriesSpec("fissions", _LABELS["fissions"]),
            ],
        )
        self._add_chart_tab(
            "環境",
            [
                SeriesSpec("season_mult", _LABELS["season_mult"]),
                SeriesSpec("hazard_mean", _LABELS["hazard_mean"]),
                SeriesSpec("hazard_coverage", _LABELS["hazard_coverage"]),
            ],
        )
        self._add_chart_tab(
            "追加行動",
            [
                SeriesSpec("dash_active", _LABELS["dash_active"]),
                SeriesSpec("defend_active", _LABELS["defend_active"]),
                SeriesSpec("rest_active", _LABELS["rest_active"]),
                SeriesSpec("mean_speed_frac", _LABELS["mean_speed_frac"]),
                SeriesSpec("mean_thrust", _LABELS["mean_thrust"]),
                SeriesSpec("mean_turn", _LABELS["mean_turn"]),
            ],
        )
        self._add_table_tab()

        self.setCentralWidget(central)

    def _add_chart_tab(self, title: str, series: list[SeriesSpec]) -> None:
        container = QWidget()
        hbox = QHBoxLayout(container)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("表示する項目"))
        lst = QListWidget()
        lst.setMinimumWidth(220)
        left_layout.addWidget(lst, stretch=1)

        btn_row = QHBoxLayout()
        all_btn = QPushButton("全選択")
        none_btn = QPushButton("全解除")
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        left_layout.addLayout(btn_row)

        view = QChartView()
        view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        view.setChart(self._build_chart(title=title))

        hbox.addWidget(left, stretch=0)
        hbox.addWidget(view, stretch=1)

        self._tabs.addTab(container, title)
        self._chart_views[title] = view
        self._chart_lists[title] = lst
        self._chart_series_specs[title] = list(series)

        def set_all(checked: bool) -> None:
            lst.blockSignals(True)
            try:
                for i in range(lst.count()):
                    item = lst.item(i)
                    if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            finally:
                lst.blockSignals(False)
            self._refresh_single_chart(title)

        all_btn.clicked.connect(lambda: set_all(True))
        none_btn.clicked.connect(lambda: set_all(False))
        lst.itemChanged.connect(lambda _item: self._refresh_single_chart(title))

    def _add_table_tab(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        vbox.addWidget(self._table)
        self._tabs.addTab(container, "テーブル")

    def _build_chart(self, *, title: str) -> QChart:
        chart = QChart()
        chart.setTitle(title)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        x_axis = QValueAxis()
        x_axis.setTitleText(_LABELS.get("tick", "Tick"))
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart._x_axis = x_axis  # type: ignore[attr-defined]
        return chart

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "統計CSVを開く",
            str(Path.cwd()),
            "CSV (*.csv);;すべてのファイル (*)",
        )
        if not path:
            return
        self.load_csv(Path(path))

    def load_csv(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            QMessageBox.critical(self, "読み込みに失敗しました", f"ファイルが見つかりません:\n{resolved}")
            return

        try:
            rows, columns = self._read_csv(resolved)
        except Exception as exc:
            QMessageBox.critical(self, "読み込みに失敗しました", f"CSVの読み込みに失敗しました:\n{exc}")
            return

        self._path = resolved
        self._rows = rows
        self._columns = columns
        self._path_label.setText(str(resolved))
        self._refresh_views()

    def _read_csv(self, path: Path) -> tuple[list[dict[str, float]], list[str]]:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return [], []
            columns = list(reader.fieldnames)
            out: list[dict[str, float]] = []
            for row in reader:
                parsed: dict[str, float] = {}
                for key in columns:
                    value = row.get(key, "")
                    parsed[key] = _to_float(value) if value is not None else 0.0
                out.append(parsed)
            return out, columns

    def _refresh_views(self) -> None:
        if QChartView is None:
            return
        for title in self._chart_views.keys():
            self._populate_check_list(title)
            self._refresh_single_chart(title)
        self._populate_table()

    def _populate_check_list(self, title: str) -> None:
        lst = self._chart_lists.get(title)
        specs = self._chart_series_specs.get(title, [])
        if lst is None:
            return
        lst.blockSignals(True)
        try:
            lst.clear()
            for spec in specs:
                item = QListWidgetItem(spec.label)
                item.setData(Qt.ItemDataRole.UserRole, spec.key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if spec.key in self._columns:
                    item.setCheckState(Qt.CheckState.Checked)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                lst.addItem(item)
        finally:
            lst.blockSignals(False)

    def _selected_specs(self, title: str) -> list[SeriesSpec]:
        lst = self._chart_lists.get(title)
        specs = self._chart_series_specs.get(title, [])
        if lst is None:
            return []
        enabled = {spec.key: spec for spec in specs if spec.key in self._columns}
        out: list[SeriesSpec] = []
        for i in range(lst.count()):
            item = lst.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            if key in enabled:
                out.append(enabled[key])
        return out

    def _refresh_single_chart(self, title: str) -> None:
        view = self._chart_views.get(title)
        if view is None:
            return
        self._populate_chart(view.chart(), self._selected_specs(title))

    def _populate_chart(self, chart: QChart, specs: list[SeriesSpec]) -> None:
        # Clear old series and y-axes (keep the shared x-axis).
        for series in list(chart.series()):
            chart.removeSeries(series)
        x_axis: QValueAxis = getattr(chart, "_x_axis")
        for axis in list(chart.axes()):
            if axis is x_axis:
                continue
            chart.removeAxis(axis)

        if not self._rows:
            return

        x_key = "tick"
        xs = [row.get(x_key, 0.0) for row in self._rows]
        if not xs:
            return

        x_min = min(xs)
        x_max = max(xs)
        x_axis.setRange(float(x_min), float(x_max))

        if not specs:
            return

        for idx, spec in enumerate(specs):
            if spec.key not in self._columns:
                continue
            series = QLineSeries()
            series.setName(spec.label)
            for row in self._rows:
                series.append(float(row.get(x_key, 0.0)), float(row.get(spec.key, 0.0)))
            chart.addSeries(series)
            series.attachAxis(x_axis)

            y_axis = QValueAxis()
            y_axis.setTitleText(spec.label)
            chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft if (idx % 2 == 0) else Qt.AlignmentFlag.AlignRight)
            series.attachAxis(y_axis)

            vals = [row.get(spec.key, 0.0) for row in self._rows]
            if vals:
                y_min = min(vals)
                y_max = max(vals)
                if y_min == y_max:
                    pad = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
                    y_axis.setRange(float(y_min - pad), float(y_max + pad))
                else:
                    pad = (y_max - y_min) * 0.05
                    y_axis.setRange(float(y_min - pad), float(y_max + pad))
            else:
                y_axis.setRange(0.0, 1.0)

    def _populate_table(self) -> None:
        if not hasattr(self, "_table"):
            return
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        if not self._columns:
            return
        self._table.setColumnCount(len(self._columns))
        self._table.setHorizontalHeaderLabels([_LABELS.get(c, c) for c in self._columns])
        preview_rows = self._rows[: min(500, len(self._rows))]
        self._table.setRowCount(len(preview_rows))
        for r, row in enumerate(preview_rows):
            for c, col in enumerate(self._columns):
                item = QTableWidgetItem(str(row.get(col, 0.0)))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
