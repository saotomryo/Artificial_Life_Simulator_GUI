from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    # Allow running via ``python app/gradio_app.py`` by adding repo root to sys.path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from app.core.config import SimulationConfig
from app.core.simulation_backend import NeatSimulationBackend, StubSimulationBackend
from app.gradio_controller import GradioSimulationController


def build_backend(
    backend_name: str,
    module_name: str,
    sim_file: Optional[Path],
    tick_substeps: int,
    sleep_interval: float,
):
    if backend_name == "stub":
        return StubSimulationBackend()
    return NeatSimulationBackend(
        module_name=module_name,
        module_path=sim_file,
        substeps=tick_substeps,
        sleep_interval=sleep_interval,
    )


def serialize_config(config: SimulationConfig) -> str:
    return json.dumps(config.to_dict(), indent=2)


def parse_config(raw: str) -> Tuple[Optional[SimulationConfig], Optional[str]]:
    if not raw:
        return SimulationConfig(), None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Config JSON parse error: {exc}"
    if not isinstance(data, dict):
        return None, "Config JSON must be an object at the top level."
    config = SimulationConfig()
    config.update_from_mapping(data)
    return config, None


def fields_to_config(values: Dict[str, Any]) -> SimulationConfig:
    config = SimulationConfig()
    config.world.width = float(values["world_width"])
    config.world.height = float(values["world_height"])
    config.world.time_step = float(values["world_time_step"])
    config.world.initial_population = int(values["world_initial_population"])

    config.resources.initial_food_pieces = int(values["resources_initial_food_pieces"])
    config.resources.initial_food_scale = bool(values["resources_initial_food_scale"])
    config.resources.food_spawn_rate = float(values["resources_food_spawn_rate"])
    config.resources.food_energy = float(values["resources_food_energy"])
    config.resources.decay_body_rate = float(values["resources_decay_body_rate"])
    config.resources.food_density_variation = float(values["resources_food_density_variation"])
    config.resources.food_type_energies = parse_float_list(values["resources_food_type_energies"])
    config.resources.food_type_weights = parse_float_list(values["resources_food_type_weights"])

    config.environment.season_period = float(values["environment_season_period"])
    config.environment.season_amplitude = float(values["environment_season_amplitude"])
    config.environment.hazard_strength = float(values["environment_hazard_strength"])
    config.environment.hazard_coverage = float(values["environment_hazard_coverage"])

    config.metabolism.base_cost = float(values["metabolism_base_cost"])
    config.metabolism.idle_cost = float(values["metabolism_idle_cost"])
    config.metabolism.starvation_energy = float(values["metabolism_starvation_energy"])
    config.metabolism.starvation_cost = float(values["metabolism_starvation_cost"])
    config.metabolism.move_cost_k = float(values["metabolism_move_cost_k"])
    config.metabolism.brain_cost_per_conn = float(values["metabolism_brain_cost_per_conn"])
    config.metabolism.fission_bias = float(values["metabolism_fission_bias"])
    config.metabolism.dash_vmax_mult = float(values["metabolism_dash_vmax_mult"])
    config.metabolism.dash_cost = float(values["metabolism_dash_cost"])
    config.metabolism.defend_strength = float(values["metabolism_defend_strength"])
    config.metabolism.defend_cost = float(values["metabolism_defend_cost"])
    config.metabolism.rest_base_cost_mult = float(values["metabolism_rest_base_cost_mult"])
    config.metabolism.rest_cost = float(values["metabolism_rest_cost"])

    config.brain.rays = int(values["brain_rays"])
    config.brain.sense_range = float(values["brain_sense_range"])
    config.brain.weight_sigma = float(values["brain_weight_sigma"])
    config.brain.add_connection_rate = float(values["brain_add_connection_rate"])
    config.brain.add_node_rate = float(values["brain_add_node_rate"])
    config.brain.delete_connection_rate = float(values["brain_delete_connection_rate"])
    config.brain.enable_advanced_actions = bool(values["brain_enable_advanced_actions"])

    return config


def parse_float_list(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    return [float(raw)]


def config_to_fields(config: SimulationConfig) -> Dict[str, Any]:
    return {
        "world_width": config.world.width,
        "world_height": config.world.height,
        "world_time_step": config.world.time_step,
        "world_initial_population": config.world.initial_population,
        "resources_initial_food_pieces": config.resources.initial_food_pieces,
        "resources_initial_food_scale": config.resources.initial_food_scale,
        "resources_food_spawn_rate": config.resources.food_spawn_rate,
        "resources_food_energy": config.resources.food_energy,
        "resources_decay_body_rate": config.resources.decay_body_rate,
        "resources_food_density_variation": config.resources.food_density_variation,
        "resources_food_type_energies": ",".join(str(x) for x in config.resources.food_type_energies),
        "resources_food_type_weights": ",".join(str(x) for x in config.resources.food_type_weights),
        "environment_season_period": config.environment.season_period,
        "environment_season_amplitude": config.environment.season_amplitude,
        "environment_hazard_strength": config.environment.hazard_strength,
        "environment_hazard_coverage": config.environment.hazard_coverage,
        "metabolism_base_cost": config.metabolism.base_cost,
        "metabolism_idle_cost": config.metabolism.idle_cost,
        "metabolism_starvation_energy": config.metabolism.starvation_energy,
        "metabolism_starvation_cost": config.metabolism.starvation_cost,
        "metabolism_move_cost_k": config.metabolism.move_cost_k,
        "metabolism_brain_cost_per_conn": config.metabolism.brain_cost_per_conn,
        "metabolism_fission_bias": config.metabolism.fission_bias,
        "metabolism_dash_vmax_mult": config.metabolism.dash_vmax_mult,
        "metabolism_dash_cost": config.metabolism.dash_cost,
        "metabolism_defend_strength": config.metabolism.defend_strength,
        "metabolism_defend_cost": config.metabolism.defend_cost,
        "metabolism_rest_base_cost_mult": config.metabolism.rest_base_cost_mult,
        "metabolism_rest_cost": config.metabolism.rest_cost,
        "brain_rays": config.brain.rays,
        "brain_sense_range": config.brain.sense_range,
        "brain_weight_sigma": config.brain.weight_sigma,
        "brain_add_connection_rate": config.brain.add_connection_rate,
        "brain_add_node_rate": config.brain.add_node_rate,
        "brain_delete_connection_rate": config.brain.delete_connection_rate,
        "brain_enable_advanced_actions": config.brain.enable_advanced_actions,
    }


FIELD_KEYS = [
    "world_width",
    "world_height",
    "world_time_step",
    "world_initial_population",
    "resources_initial_food_pieces",
    "resources_initial_food_scale",
    "resources_food_spawn_rate",
    "resources_food_energy",
    "resources_decay_body_rate",
    "resources_food_density_variation",
    "resources_food_type_energies",
    "resources_food_type_weights",
    "environment_season_period",
    "environment_season_amplitude",
    "environment_hazard_strength",
    "environment_hazard_coverage",
    "metabolism_base_cost",
    "metabolism_idle_cost",
    "metabolism_starvation_energy",
    "metabolism_starvation_cost",
    "metabolism_move_cost_k",
    "metabolism_brain_cost_per_conn",
    "metabolism_fission_bias",
    "metabolism_dash_vmax_mult",
    "metabolism_dash_cost",
    "metabolism_defend_strength",
    "metabolism_defend_cost",
    "metabolism_rest_base_cost_mult",
    "metabolism_rest_cost",
    "brain_rays",
    "brain_sense_range",
    "brain_weight_sigma",
    "brain_add_connection_rate",
    "brain_add_node_rate",
    "brain_delete_connection_rate",
    "brain_enable_advanced_actions",
]


def pack_field_values(values: Tuple[Any, ...]) -> Dict[str, Any]:
    return dict(zip(FIELD_KEYS, values))


def config_to_field_list(config: SimulationConfig) -> List[Any]:
    mapping = config_to_fields(config)
    return [mapping[key] for key in FIELD_KEYS]


def empty_field_updates() -> List[Any]:
    return [gr.update() for _ in FIELD_KEYS]


def normalize_path(file_like: Any) -> Optional[Path]:
    if file_like is None:
        return None
    if isinstance(file_like, (str, Path)):
        return Path(file_like)
    name = getattr(file_like, "name", None)
    if name:
        return Path(name)
    return None


def render_frame(frame: Optional[Dict[str, Any]], canvas_size: int = 640) -> Image.Image:
    img = Image.new("RGB", (canvas_size, canvas_size), (8, 8, 8))
    if not frame:
        return img
    draw = ImageDraw.Draw(img)
    world_w = float(frame.get("width", canvas_size))
    world_h = float(frame.get("height", canvas_size))
    scale = min(canvas_size / world_w, canvas_size / world_h) if world_w and world_h else 1.0
    offset_x = (canvas_size - world_w * scale) / 2.0
    offset_y = (canvas_size - world_h * scale) / 2.0

    def world_to_canvas(x: float, y: float) -> Tuple[float, float]:
        return offset_x + x * scale, offset_y + y * scale

    for food in frame.get("foods", []):
        x, y = world_to_canvas(food["x"], food["y"])
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(60, 200, 90))

    for body in frame.get("bodies", []):
        x, y = world_to_canvas(body["x"], body["y"])
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(150, 95, 60))

    for agent in frame.get("agents", []):
        x, y = world_to_canvas(agent["x"], agent["y"])
        size = float(agent.get("size", 1.0))
        energy = float(agent.get("energy", 0.0))
        r = max(2, int(3 + size * 2.5))
        color = energy_to_rgb(energy)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(20, 20, 20))

    return img


def energy_to_rgb(energy: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, (energy - 50.0) / 250.0))
    r = int(70 + 160 * t)
    g = int(160 - 70 * t)
    b = int(220 - 160 * t)
    return r, g, b


def append_log(log: str, message: str) -> str:
    lines = [line for line in (log or "").splitlines() if line.strip()]
    lines.append(message)
    if len(lines) > 200:
        lines = lines[-200:]
    return "\n".join(lines)


def init_controller(
    backend_name: str,
    module_name: str,
    tick_substeps: int,
    sleep_interval: float,
):
    backend = build_backend(backend_name, module_name, None, tick_substeps, sleep_interval)
    controller = GradioSimulationController(backend, SimulationConfig())
    field_values = config_to_field_list(controller.config)
    log = append_log("", f"Initialized backend: {backend_name}")
    status = "Stopped"
    return (controller, *field_values, log, status, [])


def apply_config(controller: GradioSimulationController, log: str, *fields: Any):
    if controller is None:
        return log, "Controller not initialized."
    values = pack_field_values(fields)
    config = fields_to_config(values)
    controller.apply_config(config)
    log = append_log(log, "Applied config.")
    return log, "Config applied."


def start_sim(controller: GradioSimulationController, log: str):
    if controller is None:
        return log, "Controller not initialized.", "Stopped"
    if controller.start():
        log = append_log(log, "Simulation started.")
    return log, "Running" if controller.running else "Stopped", "Running" if controller.running else "Stopped"


def stop_sim(controller: GradioSimulationController, log: str):
    if controller is None:
        return log, "Controller not initialized.", "Stopped"
    if controller.stop():
        log = append_log(log, "Simulation stopped.")
    return log, "Stopped", "Stopped"


def tick(controller: GradioSimulationController, history: List[Dict[str, Any]]):
    if controller is None:
        return (
            Image.new("RGB", (640, 640), (8, 8, 8)),
            0,
            0,
            0.0,
            0,
            0,
            0,
            0,
            [],
            {},
            history,
        )
    payload = controller.step_or_snapshot()
    frame = payload.get("frame")
    image = render_frame(frame)
    stats = {
        "tick": payload.get("tick"),
        "population": payload.get("population"),
        "mean_energy": payload.get("mean_energy"),
        "births": payload.get("births"),
        "deaths": payload.get("deaths"),
        "food": payload.get("food"),
        "bodies": payload.get("bodies"),
    }
    telemetry = payload.get("telemetry", {})
    if stats:
        history = list(history or [])
        history.append(stats)
        if len(history) > 5000:
            history = history[-5000:]
    telemetry_rows = [[k, v] for k, v in (telemetry or {}).items()]
    return (
        image,
        stats.get("tick"),
        stats.get("population"),
        stats.get("mean_energy"),
        stats.get("births"),
        stats.get("deaths"),
        stats.get("food"),
        stats.get("bodies"),
        telemetry_rows,
        frame,
        history,
    )




def load_stats_csv(file_like: Any):
    path = normalize_path(file_like)
    if path is None:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), "No CSV selected."
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), f"Failed to read CSV: {exc}"
    cols = list(df.columns)
    x_value = cols[0] if cols else None
    return (
        df,
        gr.update(choices=cols, value=x_value),
        gr.update(choices=cols, value=[c for c in cols if c != x_value][:3]),
        f"Loaded CSV: {path.name}",
    )


def plot_stats(df: Any, x_col: str, y_cols: List[str]):
    if df is None or not x_col or not y_cols:
        return None
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], label=col)
    ax.set_xlabel(x_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def download_stats_csv(history: List[Dict[str, Any]]):
    if not history:
        return None, "No stats history to export."
    df = pd.DataFrame(history)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fp:
        df.to_csv(fp.name, index=False)
        return fp.name, f"Exported {len(df)} rows."


def clear_stats_history():
    return [], "Stats history cleared."


def download_config(*fields: Any):
    values = pack_field_values(fields)
    config = fields_to_config(values)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
        fp.write(serialize_config(config))
        return fp.name, "Config prepared for download."


def load_config_file(file_like: Any, log: str):
    path = normalize_path(file_like)
    if path is None:
        return (*empty_field_updates(), log, "No config file selected.")
    try:
        content = path.read_text()
    except Exception as exc:
        return (*empty_field_updates(), log, f"Failed to read config: {exc}")
    config, error = parse_config(content)
    if error:
        return (*empty_field_updates(), log, error)
    field_values = config_to_field_list(config)
    log = append_log(log, f"Loaded config: {path.name}")
    return (*field_values, log, "Config loaded.")


def download_agents(controller: GradioSimulationController, log: str):
    if controller is None:
        return None, log, "Controller not initialized."
    path = controller.save_agents()
    log = append_log(log, f"Saved agents: {path.name}")
    return str(path), log, "Agents ready for download."


def upload_agents(controller: GradioSimulationController, file_like: Any, log: str):
    if controller is None:
        return log, "Controller not initialized."
    path = normalize_path(file_like)
    if path is None:
        return log, "No agent file selected."
    try:
        controller.load_agents(path)
    except Exception as exc:
        return log, f"Failed to load agents: {exc}"
    log = append_log(log, f"Loaded agents: {path.name}")
    return log, "Agents loaded."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Artificial Life Simulator (Gradio)") as demo:
        gr.Markdown("## Artificial Life Simulator (Gradio UI)")

        controller_state = gr.State()
        log_state = gr.State("")
        frame_state = gr.State()
        stats_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                backend_choice = gr.Radio(
                    choices=["neat", "stub"],
                    value="neat",
                    label="Backend",
                )
                module_name = gr.Textbox(
                    value="app.sim.neat_simulation",
                    label="Simulation module",
                )
                tick_substeps = gr.Slider(1, 10, value=1, step=1, label="Tick substeps")
                sleep_interval = gr.Slider(0.0, 0.1, value=0.01, step=0.005, label="Sleep interval (sec)")
                init_btn = gr.Button("Initialize backend")

                status_text = gr.Markdown("Stopped")

                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop")

                snapshot_btn = gr.Button("Snapshot")

                gr.Markdown("### Config")
                with gr.Accordion("World", open=False):
                    world_width = gr.Number(label="Width", value=2000.0, precision=2)
                    world_height = gr.Number(label="Height", value=2000.0, precision=2)
                    world_time_step = gr.Number(label="Time step", value=0.15, precision=3)
                    world_initial_population = gr.Number(label="Initial population", value=50, precision=0)

                with gr.Accordion("Resources", open=False):
                    resources_initial_food_pieces = gr.Number(label="Initial food pieces", value=2000, precision=0)
                    resources_initial_food_scale = gr.Checkbox(label="Initial food scale", value=True)
                    resources_food_spawn_rate = gr.Number(label="Food spawn rate", value=0.05, precision=3)
                    resources_food_energy = gr.Number(label="Food energy", value=50.0, precision=1)
                    resources_decay_body_rate = gr.Number(label="Decay body rate", value=0.995, precision=3)
                    resources_food_density_variation = gr.Number(label="Food density variation", value=0.0, precision=3)
                    resources_food_type_energies = gr.Textbox(
                        label="Food type energies (comma-separated)",
                        value="30.0,40.0,55.0,70.0,95.0",
                    )
                    resources_food_type_weights = gr.Textbox(
                        label="Food type weights (comma-separated)",
                        value="0.32,0.25,0.2,0.15,0.08",
                    )

                with gr.Accordion("Environment", open=False):
                    environment_season_period = gr.Number(label="Season period", value=1200.0, precision=1)
                    environment_season_amplitude = gr.Number(label="Season amplitude", value=0.0, precision=3)
                    environment_hazard_strength = gr.Number(label="Hazard strength", value=0.0, precision=3)
                    environment_hazard_coverage = gr.Number(label="Hazard coverage", value=0.0, precision=3)

                with gr.Accordion("Metabolism", open=False):
                    metabolism_base_cost = gr.Number(label="Base cost", value=0.35, precision=3)
                    metabolism_idle_cost = gr.Number(label="Idle cost", value=0.45, precision=3)
                    metabolism_starvation_energy = gr.Number(label="Starvation energy", value=120.0, precision=1)
                    metabolism_starvation_cost = gr.Number(label="Starvation cost", value=0.55, precision=3)
                    metabolism_move_cost_k = gr.Number(label="Move cost K", value=0.001, precision=4)
                    metabolism_brain_cost_per_conn = gr.Number(label="Brain cost per conn", value=0.0006, precision=5)
                    metabolism_fission_bias = gr.Number(label="Fission bias", value=1.0, precision=2)
                    metabolism_dash_vmax_mult = gr.Number(label="Dash vmax mult", value=1.5, precision=2)
                    metabolism_dash_cost = gr.Number(label="Dash cost", value=0.35, precision=2)
                    metabolism_defend_strength = gr.Number(label="Defend strength", value=0.6, precision=2)
                    metabolism_defend_cost = gr.Number(label="Defend cost", value=0.25, precision=2)
                    metabolism_rest_base_cost_mult = gr.Number(label="Rest base cost mult", value=0.4, precision=2)
                    metabolism_rest_cost = gr.Number(label="Rest cost", value=0.05, precision=2)

                with gr.Accordion("Brain & Mutation", open=False):
                    brain_rays = gr.Number(label="Rays", value=5, precision=0)
                    brain_sense_range = gr.Number(label="Sense range", value=180.0, precision=1)
                    brain_weight_sigma = gr.Number(label="Weight sigma", value=0.08, precision=3)
                    brain_add_connection_rate = gr.Number(label="Add connection rate", value=0.20, precision=2)
                    brain_add_node_rate = gr.Number(label="Add node rate", value=0.08, precision=2)
                    brain_delete_connection_rate = gr.Number(label="Delete connection rate", value=0.02, precision=2)
                    brain_enable_advanced_actions = gr.Checkbox(label="Enable advanced actions", value=False)

                apply_btn = gr.Button("Apply config")

                with gr.Row():
                    config_upload = gr.File(label="Load config file", file_types=[".json"])
                    config_download_btn = gr.Button("Download config")
                config_download = gr.File(label="Config download")

                gr.Markdown("### Agents")
                with gr.Row():
                    agents_upload = gr.File(label="Load agents file", file_types=[".json"])
                    agents_download_btn = gr.Button("Save agents")
                agents_download = gr.File(label="Agents download")

                gr.Markdown("### Log")
                log_box = gr.Textbox(lines=8, label="Log", interactive=False)

                status_message = gr.Markdown("")

            with gr.Column(scale=2):
                world_image = gr.Image(label="World View", type="pil")
                with gr.Row():
                    stat_tick = gr.Number(label="Tick", value=0, precision=0, interactive=False)
                    stat_population = gr.Number(label="Population", value=0, precision=0, interactive=False)
                    stat_mean_energy = gr.Number(label="Mean Energy", value=0.0, precision=2, interactive=False)
                with gr.Row():
                    stat_births = gr.Number(label="Births", value=0, precision=0, interactive=False)
                    stat_deaths = gr.Number(label="Deaths", value=0, precision=0, interactive=False)
                    stat_food = gr.Number(label="Food", value=0, precision=0, interactive=False)
                    stat_bodies = gr.Number(label="Bodies", value=0, precision=0, interactive=False)
                telemetry_table = gr.Dataframe(
                    headers=["Key", "Value"],
                    label="Telemetry",
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                )

                gr.Markdown("### Statistics CSV Viewer")
                stats_csv = gr.File(label="Load stats CSV", file_types=[".csv"])
                stats_table = gr.Dataframe(label="CSV Preview", interactive=False)
                with gr.Row():
                    x_column = gr.Dropdown(label="X column")
                    y_columns = gr.Dropdown(label="Y columns", multiselect=True)
                plot_btn = gr.Button("Plot CSV")
                stats_plot = gr.Plot(label="Stats Plot")
                stats_status = gr.Markdown("")
                gr.Markdown("### Export current stats history")
                with gr.Row():
                    stats_download_btn = gr.Button("Download stats CSV")
                    clear_stats_btn = gr.Button("Clear history")
                stats_download = gr.File(label="Stats download")
                stats_history_status = gr.Markdown("")

        timer = gr.Timer(0.2)

        config_fields = [
            world_width,
            world_height,
            world_time_step,
            world_initial_population,
            resources_initial_food_pieces,
            resources_initial_food_scale,
            resources_food_spawn_rate,
            resources_food_energy,
            resources_decay_body_rate,
            resources_food_density_variation,
            resources_food_type_energies,
            resources_food_type_weights,
            environment_season_period,
            environment_season_amplitude,
            environment_hazard_strength,
            environment_hazard_coverage,
            metabolism_base_cost,
            metabolism_idle_cost,
            metabolism_starvation_energy,
            metabolism_starvation_cost,
            metabolism_move_cost_k,
            metabolism_brain_cost_per_conn,
            metabolism_fission_bias,
            metabolism_dash_vmax_mult,
            metabolism_dash_cost,
            metabolism_defend_strength,
            metabolism_defend_cost,
            metabolism_rest_base_cost_mult,
            metabolism_rest_cost,
            brain_rays,
            brain_sense_range,
            brain_weight_sigma,
            brain_add_connection_rate,
            brain_add_node_rate,
            brain_delete_connection_rate,
            brain_enable_advanced_actions,
        ]

        init_btn.click(
            fn=init_controller,
            inputs=[backend_choice, module_name, tick_substeps, sleep_interval],
            outputs=[controller_state, *config_fields, log_state, status_text, stats_history_state],
        )
        init_btn.click(
            fn=lambda log: log,
            inputs=[log_state],
            outputs=[log_box],
        )

        start_btn.click(
            fn=start_sim,
            inputs=[controller_state, log_state],
            outputs=[log_state, status_message, status_text],
        )
        start_btn.click(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        stop_btn.click(
            fn=stop_sim,
            inputs=[controller_state, log_state],
            outputs=[log_state, status_message, status_text],
        )
        stop_btn.click(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        snapshot_btn.click(
            fn=tick,
            inputs=[controller_state, stats_history_state],
            outputs=[
                world_image,
                stat_tick,
                stat_population,
                stat_mean_energy,
                stat_births,
                stat_deaths,
                stat_food,
                stat_bodies,
                telemetry_table,
                frame_state,
                stats_history_state,
            ],
        )

        apply_btn.click(
            fn=apply_config,
            inputs=[controller_state, log_state, *config_fields],
            outputs=[log_state, status_message],
        )
        apply_btn.click(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        config_upload.change(
            fn=load_config_file,
            inputs=[config_upload, log_state],
            outputs=[*config_fields, log_state, status_message],
        )
        config_upload.change(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        config_download_btn.click(
            fn=download_config,
            inputs=[*config_fields],
            outputs=[config_download, status_message],
        )

        agents_download_btn.click(
            fn=download_agents,
            inputs=[controller_state, log_state],
            outputs=[agents_download, log_state, status_message],
        )
        agents_download_btn.click(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        agents_upload.change(
            fn=upload_agents,
            inputs=[controller_state, agents_upload, log_state],
            outputs=[log_state, status_message],
        )
        agents_upload.change(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

        timer.tick(
            fn=tick,
            inputs=[controller_state, stats_history_state],
            outputs=[
                world_image,
                stat_tick,
                stat_population,
                stat_mean_energy,
                stat_births,
                stat_deaths,
                stat_food,
                stat_bodies,
                telemetry_table,
                frame_state,
                stats_history_state,
            ],
        )


        stats_csv.change(
            fn=load_stats_csv,
            inputs=[stats_csv],
            outputs=[stats_table, x_column, y_columns, stats_status],
        )

        plot_btn.click(
            fn=plot_stats,
            inputs=[stats_table, x_column, y_columns],
            outputs=[stats_plot],
        )

        stats_download_btn.click(
            fn=download_stats_csv,
            inputs=[stats_history_state],
            outputs=[stats_download, stats_history_status],
        )

        clear_stats_btn.click(
            fn=clear_stats_history,
            inputs=[],
            outputs=[stats_history_state, stats_history_status],
        )

        demo.load(
            fn=init_controller,
            inputs=[backend_choice, module_name, tick_substeps, sleep_interval],
            outputs=[controller_state, *config_fields, log_state, status_text, stats_history_state],
        )
        demo.load(fn=lambda log: log, inputs=[log_state], outputs=[log_box])

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
