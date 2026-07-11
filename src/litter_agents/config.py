from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import zenoh
from pydantic_settings import BaseSettings, SettingsConfigDict

from litter_detector.config import TOPICS, Topics

_REPO_ROOT = Path(__file__).parents[2]  # src/litter_agents -> src -> repo root


def repo_path(path: str | Path) -> Path:
    """Resolve a settings path: relative ones hang off the repo root, not the CWD.

    Entry points get launched from wherever the user happens to be standing; a
    path that silently means something different per CWD is a footgun.
    """
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p

# Topics owned by the robodog-digipro stack. Kept as plain constants (not
# settings) because they are wire contracts fixed by that project — see
# robodog-digipro/src/interfaces/topics/topics.py.
ROBODOG_POSE_TOPIC = "robodog/localization/pose"
NAV_REQUEST_TOPIC = "nav/request"
NAV_STATUS_TOPIC = "nav/status"
OCCUPANCY_TOPIC = "robodog/map/occupancy"


def build_zenoh_config() -> zenoh.Config:
    """Zenoh client config pointing at the shared router."""
    endpoint = os.environ.get("ZENOH_ROUTER_ENDPOINT", "tcp/127.0.0.1:7447")
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    return cfg


class AgentSettings(BaseSettings):
    # Load secrets/overrides from the .env file at the repo root. Real environment
    # variables still take precedence over .env values. The path is absolute on
    # purpose: a bare ".env" is resolved against the CWD, so launching from
    # anywhere but the repo root would silently drop MAP_SOURCE, OLLAMA_API_KEY
    # and friends back to their defaults.
    model_config = SettingsConfigDict(env_file=_REPO_ROOT / ".env", extra="ignore")

    # ── Static map ──────────────────────────────────────────────────────────
    map_yaml_path: str = "my_lab_grid.yaml"
    map_source: Literal["file", "zenoh", "mola"] = "file"
    # MOLA SLAM control API (robodog-digipro mola_docker) — serves the static
    # map's build-grid costmap (PNG + map_server YAML) over REST on :8088.
    mola_api_url: str = "http://localhost:8088"
    mola_map_session: str = ""  # empty → the most recently modified session
    mola_build_grid: bool = False  # POST build-grid if the costmap isn't ready
    # build-grid projection window (metres, map frame): keep floor + obstacles
    # up to head height, drop the ceiling.
    mola_grid_floor_z: float = 0.0
    mola_grid_min_h: float = 0.1
    mola_grid_max_h: float = 1.5

    # ── Robot & camera geometry ─────────────────────────────────────────────
    robot_radius_m: float = 0.25  # Go2 half-width ~0.16 m + margin
    camera_fov_deg: float = 70.0
    # Distance at which the detector still reliably spots litter; beyond it a
    # cell does not count as "seen". Together with the near blind spot this
    # makes the visible region an annular wedge.
    camera_range_m: float = 3.0
    camera_min_range_m: float = 0.3

    # ── Coverage tracking ───────────────────────────────────────────────────
    coverage_update_hz: float = 5.0
    n_fov_rays: int = 90

    # ── Candidate generation & scoring ──────────────────────────────────────
    n_candidate_directions: int = 36
    candidate_step_m: float = 0.5
    candidate_min_dist_m: float = 0.5
    candidate_max_dist_m: float = 8.0
    # Gains are in m² so the weights survive a map-resolution change. For
    # scale: a completely fresh 70°/2.5 m wedge is ~3.8 m².
    w_gain: float = 1.0
    w_dist: float = 0.25  # m² of gain a meter of travel must be worth
    w_turn: float = 0.3  # m² of gain a radian of heading change must be worth
    min_gain_m2: float = 0.15
    no_gain_replans_before_stop: int = 3
    coverage_target_fraction: float = 0.95
    # Frontier-seeking fallback: when no straight-line move gains enough, walk
    # (multi-leg) toward the nearest unseen reachable cell instead of stopping.
    enable_frontier_fallback: bool = True
    frontier_blacklist_radius_m: float = 0.5

    # ── Next-best-view planner ───────────────────────────────────────────────
    planner_mode: Literal["greedy", "nbv"] = "nbv"
    n_candidates: int = 16
    candidate_min_separation_m: float = 0.5
    candidate_min_step_m: float = 0.4
    nbv_rotate_in_place: bool = True
    lambda_cost: float = 0.4
    gamma_heading: float = 0.3
    cluster_hysteresis: float = 0.25
    min_cluster_cells: int = 5
    standoff_frac_min: float = 0.4
    standoff_frac_max: float = 0.9
    frontier_bias: float = 0.8
    # Scoring rays can be coarser than coverage rays — only approximate
    # counts are needed to rank candidates.
    n_scoring_rays: int = 45

    # ── Navigation ──────────────────────────────────────────────────────────
    nav_max_speed: float = 0.4
    nav_allowed_deviation: float = 0.2
    # No nav/status at all for this long → nav stack presumed dead.
    nav_status_timeout_s: float = 5.0
    # Per-goal timeout = max(20 s, factor × distance / max_speed).
    nav_goal_timeout_factor: float = 4.0
    # After BLOCKED the robodog executor retreats toward the last reached
    # waypoint; give it time to clear the obstacle before replanning.
    blocked_retreat_wait_s: float = 2.5
    blacklist_radius_m: float = 0.5

    # ── Mission safety caps ─────────────────────────────────────────────────
    mission_max_duration_s: float = 1800.0
    mission_max_waypoints: int = 200

    # ── Debug rendering ─────────────────────────────────────────────────────
    # Drop path-planning debug frames (map + coverage + trajectory) under
    # runs/missions/<id>/debug/, mirroring the offline sim. A final overview
    # is always written; set the interval to 0 to disable the periodic frames.
    debug_render: bool = True
    debug_render_interval_s: float = 2.0

    # ── Detection validation worker ─────────────────────────────────────────
    validation_min_observations: int = 10
    validation_min_bbox_px: int = 32
    validation_min_area_px: int = 400
    validation_border_margin_px: int = 4
    validation_crop_pad: float = 0.15
    validation_queue_size: int = 16
    validation_concurrency: int = 2
    # Also send the full frame (downscaled, bbox drawn) so the model sees the
    # crop in context — cuts false validations from texture patches.
    validation_send_context: bool = True
    llm_timeout_s: float = 60.0
    llm_retries: int = 2
    llm_retry_backoff_s: float = 5.0
    findings_db_path: str = "runs/findings.db"
    findings_dir: str = "runs/missions"

    # ── MLflow tracing ──────────────────────────────────────────────────────
    # Autolog each pydantic-ai agent call (prompt, response, tokens, retries,
    # parsed output) as an MLflow trace. Empty URI → the repo-root mlflow.db
    # that training also writes to.
    mlflow_tracing: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "litter-agents"

    # ── LLM / Ollama Cloud ──────────────────────────────────────────────────
    ollama_base_url: str = "https://ollama.com/v1"
    ollama_api_key: str = ""  # env: OLLAMA_API_KEY
    vision_model_name: str = "gemma4:31b"
    text_model_name: str = "gemma4:31b"
    # Ollama Cloud does not support NativeOutput (json_schema enforcement);
    # "prompted" is the most robust mode on open vision models.
    agent_output_mode: Literal["prompted", "tool"] = "prompted"

    @staticmethod
    def topics() -> Topics:
        """Detection-pipeline topics (owned by litter_detector)."""
        return TOPICS

    @staticmethod
    def zenoh_config() -> zenoh.Config:
        return build_zenoh_config()
