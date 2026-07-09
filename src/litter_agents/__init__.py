"""Multi-agent litter-search system.

Composes the litter_detector pipeline and the robodog-digipro control stack
(both reached over Zenoh) into autonomous search missions: an LLM agent parses
the requested search area, a deterministic exploration planner sweeps it, and
a vision agent validates detected litter into a findings database.
"""
