# swarm

Python tools for the gated AI Village dataset (`aidigestorg/ai-village` on Hugging Face).
See README.md for usage and HANDOFF.md for current status and next steps. Read HANDOFF.md
first.

- Tooling: `uv sync`, `uv run pytest`, `uv run ruff check . && uv run ruff format .`
- Auth: `HF_TOKEN` env var. Data lands in `data/` (git-ignored). Never commit dataset contents.
