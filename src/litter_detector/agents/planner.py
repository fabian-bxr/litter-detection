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
from litter_detector.agents.models import Candidate
from litter_detector.agents.nbv.candidates import sample_candidate_positions
from litter_detector.agents.nbv.clusters import (
    find_frontier_clusters,
    pick_active_cluster,
)
from litter_detector.agents.nbv.geometry import polygon_area_m2, polygon_mask
from litter_detector.agents.nbv.scoring import best_candidate, score_candidates
from litter_detector.agents.nbv.seen_mask import SeenMask, rebind_bool_mask
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
        # Grid-clipped polygon mask, recomputed each ingest so it tracks
        # grid growth. Used as the second factor for occupied-in-polygon.
        self._polygon_mask: np.ndarray | None = None
        # Sticky-occupancy memory: once a cell has been observed as occupied
        # we keep it excluded from the target. Costmap glitches that briefly
        # turn an obstacle back to unknown/free won't re-introduce it as
        # unseen target area mid-mission.
        self._occupied_ever: np.ndarray | None = None
        # Total cells the polygon would occupy at grid resolution. Used as
        # the coverage denominator so cells of the polygon that lie *outside*
        # the current grid extent still count as unseen — otherwise the
        # mission completes against a slice of the polygon, not the polygon.
        self._polygon_total_cells: int = 0
        self._status = MissionStatus(mission_id="", state="idle")
        self._consecutive_nav_failures = 0
        self._max_consecutive_nav_failures = 5
        self._coverage_at_last_failure = 0.0
        # Active-cluster state (frontier clustering + commit/hysteresis)
        self._active_cluster_centroid: tuple[float, float] | None = None
        # Last-iteration debug snapshot data, read by the periodic publisher
        self._last_candidates: list[Candidate] = []
        self._last_chosen: Candidate | None = None
        self._debug_thread: threading.Thread | None = None
        self._debug_stop = threading.Event()

    # ----- public API ---------------------------------------------------

    def status(self) -> MissionStatus:
        with self._lock:
            return self._status.model_copy()

    def abort(self) -> None:
        self._abort.set()
        self.stop_periodic_debug()
        with self._lock:
            if self._status.state in ("planning", "navigating"):
                self._status = self._status.model_copy(
                    update={"state": "aborted", "last_message": "aborted by request"}
                )

    def stop_periodic_debug(self) -> None:
        self._debug_stop.set()
        if self._debug_thread is not None:
            self._debug_thread.join(timeout=1.0)
            self._debug_thread = None

    def start(self, area: SearchArea) -> MissionStatus:
        """Initialize mission state and bake the coverage target mask."""
        self._abort.clear()
        grid = self._occ.get(timeout=2.0)
        if grid is None:
            return self._set_failed("no occupancy grid available")
        self._area = area
        self._seen = SeenMask(grid)
        self._occupied_ever = grid.occupied_mask().copy()
        self._polygon_mask = polygon_mask(grid, area.polygon)
        self._target_mask = self._polygon_mask & ~self._occupied_ever
        # World-frame polygon area, in cells, fixed for the mission. Survives
        # grid growth so coverage stays consistent as the costmap expands.
        self._polygon_total_cells = max(
            1, int(round(polygon_area_m2(area.polygon) / (grid.resolution ** 2)))
        )
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
        self._consecutive_nav_failures = 0
        self._coverage_at_last_failure = 0.0
        self._active_cluster_centroid = None
        self._last_candidates = []
        self._last_chosen = None
        # Kick off periodic debug publisher (~2 Hz) for the lifetime of the mission.
        if self._debug_publisher is not None:
            self._debug_stop = threading.Event()
            self._debug_thread = threading.Thread(
                target=self._periodic_debug_loop,
                daemon=True,
                name="planner-debug-publish",
            )
            self._debug_thread.start()
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
            coverage = self._coverage_locked()
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

        # Frontier clustering with commit + hysteresis: identify connected
        # unseen regions, pick one to commit to, and bias candidate sampling
        # toward its cells. This is what stops the wander/back-jump pattern.
        clusters = find_frontier_clusters(
            unseen, grid, min_size=self._params.min_cluster_cells
        )
        active, active_centroid = pick_active_cluster(
            clusters, pose, self._active_cluster_centroid, self._params
        )
        self._active_cluster_centroid = active_centroid
        sampling_mask = active.cells if active is not None else unseen

        positions = sample_candidate_positions(
            pose, grid, self._area.polygon, self._params, self._rng, safe,
            unseen_mask=sampling_mask,
        )
        candidates = score_candidates(pose, positions, grid, self._params, unseen)
        choice = best_candidate(candidates)
        with self._lock:
            self._last_candidates = list(candidates)
            self._last_chosen = choice
        if choice is None:
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
            # Soft retry: the streamer kept the seen mask up-to-date during
            # motion, so progress made before the failure is preserved. We
            # replan from the current pose. Reset the strike counter whenever
            # coverage has grown since the last failure — glitchy nav on a
            # mission that's still progressing shouldn't ever terminate.
            with self._seen_lock:
                coverage = self._coverage_locked()
            if coverage > self._coverage_at_last_failure + 1e-6:
                self._consecutive_nav_failures = 0
            self._consecutive_nav_failures += 1
            self._coverage_at_last_failure = coverage
            # Drop active cluster — current target may be unreachable; let
            # next iteration re-pick based on what's actually accessible.
            self._active_cluster_centroid = None
            reason = (
                "nav timed out" if nav_result is None else f"nav state={nav_result.state}"
            )
            if (
                self._consecutive_nav_failures
                >= self._max_consecutive_nav_failures
            ):
                return self._set_state(
                    state="failed",
                    coverage=coverage,
                    iteration=iteration,
                    message=f"{reason} ({self._consecutive_nav_failures}x without progress); giving up",
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
            coverage = self._coverage_locked()
        grid = final_grid
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

    def _coverage_locked(self) -> float:
        """Coverage as fraction of the polygon's *world* area, not the grid slice.

        Numerator: cells that are seen AND in the grid-clipped target.
        Denominator: `_polygon_total_cells` (fixed-from-world-area) minus
        cells we've ever observed as occupied inside the polygon — so
        revealed obstacles correctly shrink the goal, but unexplored polygon
        cells *outside the current grid extent* still count as unseen.
        Caller must hold `_seen_lock`.
        """
        if (
            self._seen is None
            or self._target_mask is None
            or self._occupied_ever is None
            or self._polygon_mask is None
        ):
            return 0.0
        seen_inside = int((self._seen.mask & self._target_mask).sum())
        occupied_in_poly = int((self._occupied_ever & self._polygon_mask).sum())
        denom = max(1, self._polygon_total_cells - occupied_in_poly)
        return min(1.0, seen_inside / denom)

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
                old_res = self._seen.resolution
                old_ox = self._seen.origin_x
                old_oy = self._seen.origin_y
                self._seen.rebind(grid)
                if self._occupied_ever is not None:
                    self._occupied_ever = rebind_bool_mask(
                        self._occupied_ever, old_res, old_ox, old_oy, grid
                    )
            if self._occupied_ever is None:
                self._occupied_ever = np.zeros(
                    (grid.height, grid.width), dtype=bool
                )
            # Sticky: OR in any newly-observed occupied cells; never clear.
            self._occupied_ever |= grid.occupied_mask()
            self._polygon_mask = polygon_mask(grid, self._area.polygon)
            self._target_mask = self._polygon_mask & ~self._occupied_ever
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

    def _periodic_debug_loop(self) -> None:
        """Render and publish a debug JPEG at ~2 Hz throughout the mission.

        Decoupled from the step() cycle: the user gets a live view of the
        seen mask growing as the robot drives, instead of one frame per nav
        iteration. Reads the latest pose+grid each tick and snapshots the
        seen/target masks under `_seen_lock`.
        """
        if self._debug_publisher is None:
            return
        from litter_detector.agents.nbv.visualize import render_debug_jpeg

        period_s = 0.5
        while not self._debug_stop.is_set():
            try:
                if (
                    self._seen is None
                    or self._target_mask is None
                    or self._area is None
                ):
                    self._debug_stop.wait(period_s)
                    continue
                pose = self._pose.get(timeout=0.2)
                grid = self._occ.get(timeout=0.2)
                if pose is None or grid is None:
                    self._debug_stop.wait(period_s)
                    continue
                with self._seen_lock:
                    if not self._seen.matches(grid):
                        # Geometry changed mid-tick; let the streamer/step
                        # rebind on the next iteration.
                        self._debug_stop.wait(period_s)
                        continue
                    seen_snapshot = self._seen.mask.copy()
                    target_snapshot = self._target_mask.copy()
                with self._lock:
                    candidates = list(self._last_candidates)
                    chosen = self._last_chosen
                    iteration = self._status.iteration
                    coverage = self._status.coverage
                jpeg = render_debug_jpeg(
                    grid=grid,
                    seen_mask=seen_snapshot,
                    target_mask=target_snapshot,
                    polygon=self._area.polygon,
                    pose=pose,
                    candidates=candidates,
                    chosen=chosen,
                    iteration=iteration,
                    coverage=coverage,
                )
                self._debug_publisher.put(jpeg)
            except Exception as e:
                logger.debug(f"periodic debug skipped: {e}")
            self._debug_stop.wait(period_s)

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
        try:
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
                    logger.info(
                        f"Planner terminal: {status.state} — {status.last_message}"
                    )
                    return
        finally:
            self._core.stop_periodic_debug()

    def abort(self) -> None:
        self._core.abort()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> MissionStatus:
        return self._core.status()
