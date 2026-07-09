"""Pydantic-AI agents: search-area parsing, vision validation, report summary.

All agents run against Ollama Cloud (or any OpenAI-compatible Ollama endpoint)
and are leaf functions — the mission control flow is plain asyncio, agents are
only called at fixed decision points.
"""
