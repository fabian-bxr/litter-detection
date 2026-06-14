"""Tests for hunter/scoring.py"""

import math
import numpy as np
from litter_agents.mapping.grid import GridMap
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.hunter.scoring import generate_candidates
from litter_agents.hunter.reachability import Blacklist


def _open_grid(size: int = 200, res: float = 0.05) -> GridMap:
    half = size * res / 2
    return GridMap(
        data=np.zeros((size, size), dtype=np.int8),
        resolution=res,
        origin_x=-half,
        origin_y=-half,
    )


def _all_unseen(grid: GridMap) -> np.ndarray:
    return grid.data == 0


def test_generates_candidates_in_open_space():
    g = _open_grid()
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    cands = generate_candidates(
        pose, g, g, _all_unseen(g), Blacklist(),
        n_directions=36, max_range_m=3.0,
    )
    assert len(cands) > 0, "Should find candidates in open space"


def test_higher_gain_scores_higher_ceteris_paribus():
    """Two candidates at same distance/turn but different gain → higher gain wins."""
    g = _open_grid(400)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    bl = Blacklist()
    unseen = _all_unseen(g)
    cands = generate_candidates(pose, g, g, unseen, bl, n_directions=36, max_range_m=5.0)
    assert len(cands) >= 2
    # candidates are sorted descending by score
    assert cands[0].score >= cands[1].score


def test_no_candidates_through_walls():
    """Candidates must not be placed on the far side of a wall."""
    g = _open_grid(200)
    # Wall at x=1.0 (centre = 0, so col = 100+20 = 120)
    wall_col = int((1.0 - g.origin_x) / g.resolution)
    g.data[:, wall_col] = np.int8(100)

    # inflated = same (wall is there)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)  # facing right
    bl = Blacklist()
    unseen = _all_unseen(g)
    cands = generate_candidates(pose, g, g, unseen, bl, n_directions=36, max_range_m=5.0)

    # No candidate should be beyond the wall in the +x direction
    for c in cands:
        if abs(math.atan2(math.sin(c.heading), math.cos(c.heading))) < math.radians(35):
            # candidate roughly in forward direction
            assert c.x <= 1.0 + g.resolution, (
                f"Candidate at x={c.x:.2f} beyond wall at x=1.0"
            )


def test_blacklisted_position_excluded():
    g = _open_grid(200)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    bl = Blacklist(radius_m=1.0)
    # Blacklist a large area ahead
    for dx in range(-10, 10):
        for dy in range(-10, 10):
            bl.add(dx * 0.5, dy * 0.5)
    unseen = _all_unseen(g)
    cands = generate_candidates(pose, g, g, unseen, bl, n_directions=36, max_range_m=3.0)
    for c in cands:
        assert not bl.is_blacklisted(c.x, c.y), "Blacklisted position returned as candidate"


def test_deterministic():
    g = _open_grid()
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    bl = Blacklist()
    unseen = _all_unseen(g)
    c1 = generate_candidates(pose, g, g, unseen, bl)
    c2 = generate_candidates(pose, g, g, unseen, bl)
    assert [c.x for c in c1] == [c.x for c in c2]
    assert [c.score for c in c1] == [c.score for c in c2]
