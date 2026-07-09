from __future__ import annotations

from pathlib import Path

from litter_detector.tracker.registry import ObjectRegistry
from litter_detector.tracker.types import BBox, Track


def _track(id_: int = 1, first_ns: int = 100, last_ns: int = 200, n_obs: int = 5) -> Track:
    return Track(
        id=id_,
        bbox=BBox(10, 20, 30, 40),
        area_px=600,
        first_seen_ns=first_ns,
        last_seen_ns=last_ns,
        n_observations=n_obs,
    )


def test_upsert_then_get_roundtrips(tmp_path: Path) -> None:
    reg = ObjectRegistry(tmp_path / "objects.db")
    try:
        t = _track()
        reg.upsert(t)
        assert reg.get(1) == t
    finally:
        reg.close()


def test_upsert_updates_last_seen_but_preserves_first_seen(tmp_path: Path) -> None:
    reg = ObjectRegistry(tmp_path / "objects.db")
    try:
        reg.upsert(_track(first_ns=100, last_ns=200, n_obs=3))
        reg.upsert(_track(first_ns=100, last_ns=400, n_obs=8))
        stored = reg.get(1)
        assert stored is not None
        assert stored.first_seen_ns == 100
        assert stored.last_seen_ns == 400
        assert stored.n_observations == 8
    finally:
        reg.close()


def test_upsert_all_persists_multiple_tracks(tmp_path: Path) -> None:
    reg = ObjectRegistry(tmp_path / "objects.db")
    try:
        reg.upsert_all([_track(id_=1), _track(id_=2), _track(id_=3)])
        assert {t.id for t in reg.all()} == {1, 2, 3}
    finally:
        reg.close()


def test_upsert_all_with_empty_list_is_noop(tmp_path: Path) -> None:
    reg = ObjectRegistry(tmp_path / "objects.db")
    try:
        reg.upsert_all([])
        assert reg.all() == []
    finally:
        reg.close()


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    reg = ObjectRegistry(tmp_path / "objects.db")
    try:
        assert reg.get(999) is None
    finally:
        reg.close()


def test_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "objects.db"
    reg = ObjectRegistry(nested)
    try:
        assert nested.exists()
    finally:
        reg.close()
