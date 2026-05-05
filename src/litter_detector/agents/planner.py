from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from loguru import logger

from litter_detector.agents.models import (
    MissionState,
    MissionStatus,
    NBVParams,
    Pose,
    SearchArea,
)
from litter_detector.agents.nbv.candidates import sample_candidate_positions
from litter_detector.agents.nbv.geometry import polygon_mask
from litter_detector.agents.nbv.scoring import best_candidate, score_candidates
from litter_detector.agents.nbv.seen_mask import SeenMask
from litter_detector.agents.tools.nav import NavResult
from litter_detector.agents.tools.occupancy import OccupancyGrid


class _PoseSrc(Protocol):
    def get(self, timeout: float = 1.0) -> Pose | None: ...


class _OccSrc(Protocol):
    def get(self, timeout: float = 1.0) -> OccupancyGrid | None: ...


class _NavSink(Protocol):
    def submit(self, target: Pose, /, **kwargs: object) -> str: ...

    def wait_for_terminal(
        self, request_id: str, timeout: float
    ) -> NavResult | None: ...


_NAV_TIMEOUT_S = 30.0


class PlannerCore:
    """Deterministic greedy NBV coverage loop.

    Step-based API for testability and threadability:
        core.start(area)       # initializes mission state
        core.step()            # one NBV iteration → MissionStatus
        core.status()          # latest status snapshot
        core.abort()           # request termination
    """

    def __init__(
        self,
        pose: _PoseSrc,
        occupancy: _OccSrc,
        nav: Any,
        params: NBVParams | None = None,
        rng_seed: int = 0,
        nav_timeout_s: float = _NAV_TIMEOUT_S,
        debug_dir: Path | None = None,
    ) -> None:
        self._pose = pose
        self._occ = occupancy
        self._nav = nav
        self._params = params or NBVParams()
        self._rng = np.random.default_rng(rng_seed)
        self._nav_timeout_s = nav_timeout_s
        self._debug_dir = debug_dir

        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._area: SearchArea | None = None
        self._seen: SeenMask | None = None
        self._target_mask: np.ndarray | None = None
        self._status = MissionStatus(mission_id="", state="idle")

    # ----- public API ---------------------------------------------------

    def status(self) -> MissionStatus:
        with self._lock:
            return self._status.model_copy()

    def abort(self) -> None:
        self._abort.set()
        with self._lock:
            if self._status.state in ("planning", "navigating"):
                self._status = self._status.model_copy(
                    update={"state": "aborted", "last_message": "aborted by request"}
                )

    def start(self, area: SearchArea) -> MissionStatus:
        """Initialize mission state and bake the coverage target mask."""
        self._abort.clear()
        grid = self._occ.get(timeout=2.0)
        if grid is None:
            return self._set_failed("no occupancy grid available")
        self._area = area
        self._seen = SeenMask(grid)
        self._target_mask = polygon_mask(grid, area.polygon) & grid.free_mask()
        target_count = int(self._target_mask.sum())
        if target_count == 0:
            return self._set_failed("search polygon contains no free cells")
        with self._lock:
            self._status = MissionStatus(
                mission_id=str(uuid.uuid4()),
                state="planning",
                coverage=0.0,
                iteration=0,
                last_message=f"target {target_count} cells",
            )
        logger.info(f"Planner started mission with {target_count} target cells")
        return self.status()

    def step(self) -> MissionStatus:
        """Execute one NBV iteration. Returns the resulting status."""
        if self._area is None or self._seen is None or self._target_mask is None:
            return self._set_failed("planner not started")
        if self._abort.is_set():
            return self.status()

        with self._lock:
            cur = self._status
            if cur.state in ("completed", "failed", "aborted", "blocked"):
                return cur.model_copy()
            iteration = cur.iteration + 1

        pose = self._pose.get(timeout=2.0)
        if pose is None:
            return self._set_failed("no pose available")
        grid = self._occ.get(timeout=2.0)
        if grid is None:
            return self._set_failed("no occupancy grid available")
        if not self._seen.matches(grid):
            logger.info(
                f"occupancy grid geometry changed "
                f"({self._seen.shape} @ res={self._seen.resolution} → "
                f"({grid.height},{grid.width}) @ res={grid.resolution}); rebinding"
            )
            self._seen.rebind(grid)
            self._target_mask = (
                polygon_mask(grid, self._area.polygon) & grid.free_mask()
            )

        # Always update seen mask from where we currently are.
        self._seen.update(pose, grid, self._params)

        coverage = self._seen.coverage_inside(self._target_mask)
        if coverage >= self._params.coverage_target:
            return self._set_state(
                state="completed",
                coverage=coverage,
                iteration=iteration,
                message=f"coverage target reached ({coverage:.2%})",
            )
        if iteration > self._params.max_iterations:
            return self._set_state(
                state="failed",
                coverage=coverage,
                iteration=iteration,
                message=f"max iterations ({self._params.max_iterations}) reached",
            )

        unseen = self._target_mask & ~self._seen.mask
        safe = grid.inflate_obstacles(self._params.obstacle_inflation_m)
        positions = sample_candidate_positions(
            pose, grid, self._area.polygon, self._params, self._rng, safe
        )
        candidates = score_candidates(pose, positions, grid, self._params, unseen)
        choice = best_candidate(candidates)
        if choice is None:
            self._maybe_render_debug(
                grid=grid,
                polygon=self._area.polygon,
                pose=pose,
                candidates=candidates,
                chosen=None,
                iteration=iteration,
                coverage=coverage,
            )
            return self._set_state(
                state="blocked",
                coverage=coverage,
                iteration=iteration,
                message="no candidate offers positive gain",
            )

        self._set_state(
            state="navigating",
            coverage=coverage,
            iteration=iteration,
            message=f"→ ({choice.pose.x:.2f},{choice.pose.y:.2f}) gain={choice.gain:.2%}",
            current_target=choice.pose,
        )
        request_id = self._nav.submit(choice.pose)
        nav_result = self._nav.wait_for_terminal(request_id, timeout=self._nav_timeout_s)

        if nav_result is None:
            return self._set_state(
                state="failed",
                coverage=coverage,
                iteration=iteration,
                message="nav timed out",
            )
        if nav_result.state == "blocked":
            return self._set_state(
                state="blocked",
                coverage=coverage,
                iteration=iteration,
                message="nav reported blocked",
            )
        if nav_result.state != "arrived_final":
            return self._set_state(
                state="failed",
                coverage=coverage,
                iteration=iteration,
                message=f"nav state={nav_result.state}",
            )

        # Re-cast FOV from the actual final pose.
        final_pose = nav_result.final_pose if nav_result.final_pose is not None else pose
        if nav_result.final_pose is not None:
            self._seen.update(nav_result.final_pose, grid, self._params)
        coverage = self._seen.coverage_inside(self._target_mask)
        self._maybe_render_debug(
            grid=grid,
            polygon=self._area.polygon,
            pose=final_pose,
            candidates=candidates,
            chosen=choice,
            iteration=iteration,
            coverage=coverage,
        )
        if self._abort.is_set():
            return self._set_state(
                state="aborted",
                coverage=coverage,
                iteration=iteration,
                message="aborted by request",
            )
        new_state = (
            "completed" if coverage >= self._params.coverage_target else "planning"
        )
        return self._set_state(
            state=new_state,
            coverage=coverage,
            iteration=iteration,
            message=f"arrived; coverage={coverage:.2%}",
        )

    # ----- helpers ------------------------------------------------------

    def _set_state(
        self,
        *,
        state: MissionState,
        coverage: float,
        iteration: int,
        message: str,
        current_target: Pose | None = None,
    ) -> MissionStatus:
        with self._lock:
            self._status = MissionStatus(
                mission_id=self._status.mission_id or str(uuid.uuid4()),
                state=state,
                coverage=coverage,
                iteration=iteration,
                last_message=message,
                current_target=current_target,
            )
            return self._status.model_copy()

    def _maybe_render_debug(
        self,
        *,
        grid,
        polygon,
        pose: Pose,
        candidates: list,
        chosen,
        iteration: int,
        coverage: float,
    ) -> None:
        if self._debug_dir is None or self._seen is None or self._target_mask is None:
            return
        try:
            from litter_detector.agents.nbv.visualize import render_debug_png

            mid = self._status.mission_id or "mission"
            out = self._debug_dir / mid / f"iter_{iteration:03d}.png"
            render_debug_png(
                out_path=out,
                grid=grid,
                seen_mask=self._seen.mask,
                target_mask=self._target_mask,
                polygon=polygon,
                pose=pose,
                candidates=candidates,
                chosen=chosen,
                iteration=iteration,
                coverage=coverage,
            )
        except Exception as e:
            logger.warning(f"debug render failed: {e}")

    def _set_failed(self, message: str) -> MissionStatus:
        with self._lock:
            self._status = MissionStatus(
                mission_id=self._status.mission_id or str(uuid.uuid4()),
                state="failed",
                coverage=self._status.coverage,
                iteration=self._status.iteration,
                last_message=message,
            )
            return self._status.model_copy()


_TERMINAL_STATES = {"completed", "failed", "aborted", "blocked"}


class PlannerRunner:
    """Drives PlannerCore.step() in a background thread until terminal."""

    def __init__(self, core: PlannerCore) -> None:
        self._core = core
        self._thread: threading.Thread | None = None

    def start(self, area: SearchArea) -> MissionStatus:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("PlannerRunner already running")
        status = self._core.start(area)
        if status.state == "failed":
            return status
        self._thread = threading.Thread(target=self._loop, daemon=True, name="planner")
        self._thread.start()
        return status

    def _loop(self) -> None:
        while True:
            if self._core._abort.is_set():
                self._core._set_state(
                    state="aborted",
                    coverage=self._core.status().coverage,
                    iteration=self._core.status().iteration,
                    message="aborted by request",
                )
                return
            status = self._core.step()
            if status.state in _TERMINAL_STATES:
                logger.info(f"Planner terminal: {status.state} — {status.last_message}")
                return

    def abort(self) -> None:
        self._core.abort()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> MissionStatus:
        return self._core.status()
