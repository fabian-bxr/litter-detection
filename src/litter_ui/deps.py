from __future__ import annotations

from pathlib import Path

from litter_agents.config import AgentSettings
from litter_agents.validation.findings import FindingsRepository

_REPO_ROOT = Path(__file__).parents[2]  # src/litter_ui -> src -> repo root

_repo: FindingsRepository | None = None


def get_repo() -> FindingsRepository:
    global _repo
    if _repo is None:
        db_path = Path(AgentSettings().findings_db_path)
        if not db_path.is_absolute():
            db_path = _REPO_ROOT / db_path
        _repo = FindingsRepository(db_path)
    return _repo
