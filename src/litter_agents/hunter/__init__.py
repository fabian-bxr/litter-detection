"""Deterministic exploration core.

Everything in this package is pure computation over numpy grids and Pose2D —
no Zenoh, no LLMs, no I/O — so the whole search behavior can be exercised
offline (see litter_agents.sim) and unit-tested cell-exactly.
"""
