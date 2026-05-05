from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from litter_detector.agents.config import AgentsConfig
from litter_detector.agents.mission import (
    MissionDeps,
    build_default_runner,
    build_mission_agent,
    ollama_model,
)
from litter_detector.agents.tracing import setup_mlflow_tracing


async def _repl(agent, deps: MissionDeps) -> None:
    print("Mission REPL — type a search request, '/status' to poll, '/quit' to exit.")
    history = None
    while True:
        try:
            line = await asyncio.to_thread(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/status":
            print(f"status: {deps.runner.status().model_dump_json(indent=2)}")
            continue
        if line == "/abort":
            deps.runner.abort()
            print("abort requested")
            continue

        result = await agent.run(line, deps=deps, message_history=history)
        history = result.all_messages()
        print(f"agent> {result.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Litter-search mission agent REPL")
    parser.add_argument(
        "--ui",
        choices=["repl"],
        default="repl",
        help="Frontend (only 'repl' for now; AG-UI deferred)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="If set, save per-iteration debug PNGs to this directory.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracing."
    )
    args = parser.parse_args()

    if not args.no_mlflow:
        setup_mlflow_tracing()

    cfg = AgentsConfig()
    model = ollama_model(cfg.mission)
    runner, pose, occ, nav = build_default_runner(debug_dir=args.debug_dir)
    deps = MissionDeps(pose=pose, runner=runner)
    agent = build_mission_agent(model)
    logger.info(f"Mission Agent ready (model={cfg.mission.model})")

    try:
        if args.ui == "repl":
            asyncio.run(_repl(agent, deps))
    finally:
        runner.abort()
        runner.join(timeout=5.0)
        pose.close()
        occ.close()
        nav.close()


if __name__ == "__main__":
    main()
