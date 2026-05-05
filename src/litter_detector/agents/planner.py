from __future__ import annotations

import threading
import uuid
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


class _DebugSink(Protocol):
    def put(self, payload: bytes, /) -> None: ...


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
        debug_publisher: _DebugSink | None = None,
    ) -> None:
        self._pose = pose
        self._occ = occupancy
        self._nav = nav
        self._params = params or NBVParams()
        self._rng = np.random.default_rng(rng_seed)
        self._nav_timeout_s = nav_timeout_s
        self._debug_publisher = debug_publisher

        self._lock = threading.Lock()
        self._seen_lock = threading.Lock()
        self._abort = threading.Event()
        self._area: SearchArea | None = None
        self._seen: SeenMask | None = None
        self._target_mask: np.ndarray | None = None
        self._status = MissionStatus(mission_id="", state="idle")
        self._consecutive_nav_failures = 0
        self._max_consecutive_nav_failures = 3

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
        self._target_mask = polygon_mask(grid, area.polygon) & ~grid.occupied_mask()
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
        self._ingest(pose, grid)

        with self._seen_lock:
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

        with self._seen_lock:
            unseen = self._target_mask & ~self._seen.mask
        safe = grid.inflate_obstacles(self._params.obstacle_inflation_m)
        positions = sample_candidate_positions(
            pose, grid, self._area.polygon, self._params, self._rng, safe,
            unseen_mask=unseen,
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
        stop_streamer = threading.Event()
        streamer = threading.Thread(
            target=self._stream_seen_updates,
            args=(stop_streamer,),
            daemon=True,
            name="planner-seen-stream",
        )
        streamer.start()
        try:
            nav_result = self._nav.wait_for_terminal(
                request_id, timeout=self._nav_timeout_s
            )
        finally:
            stop_streamer.set()
            streamer.join(timeout=1.0)

        if nav_result is None or (
            nav_result.state != "arrived_final" and nav_result.state != "blocked"
        ):
            # Soft retry: keep the seen mask we accumulated during motion (the
            # streamer was running) and replan from the current pose. The mission
            # only fails after `_max_consecutive_nav_failures` strikes in a row.
            self._consecutive_nav_failures += 1
            with self._seen_lock:
                coverage = self._seen.coverage_inside(self._target_mask)
            reason = "nav timed out" if nav_result is None else f"nav state={nav_result.state}"
            if (
                self._consecutive_nav_failures
                >= self._max_consecutive_nav_failures
            ):
                return self._set_state(
                    state="failed",
                    coverage=coverage,
                    iteration=iteration,
                    message=f"{reason} ({self._consecutive_nav_failures}x); giving up",
                )
            logger.warning(
                f"{reason} (strike {self._consecutive_nav_failures}/"
                f"{self._max_consecutive_nav_failures}); replanning from current pose"
            )
            return self._set_state(
                state="planning",
                coverage=coverage,
                iteration=iteration,
                message=f"{reason}; replanning",
            )
        if nav_result.state == "blocked":
            self._consecutive_nav_failures = 0
            return self._set_state(
                state="blocked",
                coverage=coverage,
                iteration=iteration,
                message="nav reported blocked",
            )
        # arrived_final: clear the strike counter.
        self._consecutive_nav_failures = 0

        # Re-cast FOV from the actual final pose, with a fresh grid snapshot.
        final_pose = nav_result.final_pose if nav_result.final_pose is not None else pose
        final_grid = self._occ.get(timeout=1.0) or grid
        if nav_result.final_pose is not None:
            self._ingest(nav_result.final_pose, final_grid)
        with self._seen_lock:
            coverage = self._seen.coverage_inside(self._target_mask)
        grid = final_grid
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

    def _ingest(self, pose: Pose, grid: OccupancyGrid) -> None:
        """Atomically: rebind on geometry change, recompute target, OR FOV wedge.

        Holds `_seen_lock` so concurrent readers (status, scoring) and the
        streaming updater see a consistent (mask, target, geometry) triple.
        Target = polygon AND not-known-occupied; unknown cells count as
        to-be-covered and are marked seen as the FOV raycast sweeps them.
        """
        if self._area is None or self._seen is None:
            return
        with self._seen_lock:
            if not self._seen.matches(grid):
                logger.debug(
                    f"grid geometry changed "
                    f"({self._seen.shape} → ({grid.height},{grid.width})); rebinding"
                )
                self._seen.rebind(grid)
            self._target_mask = (
                polygon_mask(grid, self._area.polygon) & ~grid.occupied_mask()
            )
            self._seen.update(pose, grid, self._params)

    def _stream_seen_updates(self, stop: threading.Event) -> None:
        """Poll pose+grid while nav is in flight and fold them into the seen mask.

        Runs at ~10 Hz. Skips iterations where pose or grid is unavailable.
        Exits when `stop` is set (called from `step()` after nav terminates).
        """
        period_s = 0.1
        while not stop.is_set():
            if self._abort.is_set():
                return
            p = self._pose.get(timeout=period_s)
            g = self._occ.get(timeout=period_s)
            if p is not None and g is not None:
                try:
                    self._ingest(p, g)
                except Exception as e:
                    logger.warning(f"streaming seen-update failed: {e}")
            stop.wait(period_s)

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
        if (
            self._debug_publisher is None
            or self._seen is None
            or self._target_mask is None
        ):
            return
        try:
            from litter_detector.agents.nbv.visualize import render_debug_jpeg

            with self._seen_lock:
                seen_snapshot = self._seen.mask.copy()
                target_snapshot = self._target_mask.copy()
            jpeg = render_debug_jpeg(
                grid=grid,
                seen_mask=seen_snapshot,
                target_mask=target_snapshot,
                polygon=polygon,
                pose=pose,
                candidates=candidates,
                chosen=chosen,
                iteration=iteration,
                coverage=coverage,
            )
            self._debug_publisher.put(jpeg)
        except Exception as e:
            logger.warning(f"debug render/publish failed: {e}")

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
