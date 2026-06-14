from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).parent.parent.parent


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LITTER_AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # Ollama / LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = "ollama"
    ollama_text_model: str = "gemma4:31b-cloud"
    ollama_vision_model: str = "gemma4:31b-cloud"
    agent_output_mode: str = "prompted"  # "prompted" | "tool"

    # Map
    map_file: str = "my_lab_grid.yaml"
    robot_radius_m: float = 0.35

    # Camera / raycasting
    fov_deg: float = 70.0
    seen_range_m: float = 1.0
    camera_min_range_m: float = 0.3

    # Exploration scoring
    w_gain: float = 1.0
    w_dist: float = 0.25
    w_turn: float = 0.3
    sample_start_m: float = 0.3    # first sample distance along each direction
    sample_step_m: float = 0.3     # spacing between successive samples

    # Termination
    coverage_threshold: float = 0.92
    min_gain_m2: float = 0.002
    consecutive_low_gain_limit: int = 8
    mission_max_duration_s: float = 1800.0
    mission_max_waypoints: int = 200

    # Zenoh
    zenoh_router_endpoint: str = "tcp/127.0.0.1:7447"

    # Findings
    findings_db: str = "runs/findings.db"
    mission_images_dir: str = "runs/missions"

    @property
    def map_path(self) -> Path:
        p = Path(self.map_file)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def findings_db_path(self) -> Path:
        p = Path(self.findings_db)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def mission_images_path(self) -> Path:
        p = Path(self.mission_images_dir)
        return p if p.is_absolute() else REPO_ROOT / p


settings = AgentSettings()
