---
name: litter-agents-plan-decisions
description: Approved plan + user decisions for the litter_agents multi-agent search system (branch feature/litter-agents)
metadata: 
  node_type: memory
  type: project
  originSessionId: 06a21a74-8d84-4079-8467-34623b94d0ce
---

Task (from `prompts/litter-agents.md`): multi-agent litter-search system — user prompt like "Search 10m around me for litter" drives the Go2 to explore a mapped area, validate detections with an Ollama Cloud vision model, store findings in SQLite. Full approved plan: `.claude/plans/litter-agents.md` (written 2026-06-12, approved; implementation not yet started as of then).

Decisions fixed with the user (do not re-ask):
- **Exploration/path-planning is pure deterministic** (raycasting coverage + info-gain scoring + flood-fill reachability), no LLM in the loop. Pydantic-AI agents only for: search-area parsing, vision validation, optional report summary; composed by plain asyncio, not agent-as-tool.
- **Vision model: Gemma on Ollama Cloud** (configurable); on Ollama Cloud use `PromptedOutput`/`ToolOutput`, never `NativeOutput` (unsupported there).
- **Camera: FoV 70°, effective seen-range 2.5 m** for coverage raycasting.
- **`my_lab_grid.png` is MOLA mm2grid output** (ROS map_server PNG+YAML format, 405×225 px); metadata is *placeholder* (0.05 m/px, centered origin) until the real YAML arrives — user said "use placeholder for now, prepare for mm2grid format". Map later served via Zenoh/REST → `MapProvider` abstraction, robodog's `OccupancyGrid` schema as canonical representation.
- New package `src/litter_agents/`, entry points `litter-mission` + `litter-sim`; `litter_detector` untouched except one line (attach `frame_ts_ns` to `litter/frame` publication at `detector/main.py:211`).
- Findings store **robot pose + camera bearing**, not litter world position (no depth).

Robot-side wire contracts: see [[robodog-digipro-wire-contracts]]. User pre-created empty `tests/agents/` and `tests/hunter/` dirs — the test layout maps onto them.
