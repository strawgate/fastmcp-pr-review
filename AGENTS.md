# Agent Instructions

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for architecture, data flow, and design decisions.

See [DEVELOPING.md](DEVELOPING.md) for build/test commands, sampling patterns, and how to add new review modes.

## Agent-Specific Rules

1. **Each mode file is self-contained.** Prompts and stage models are defined inline. A reader should understand the complete pattern from one file.
2. **`fast.py` and `thorough.py` must not import from each other.** They share only `models.py` types.
3. **Prompts use general principles, not specific pattern exclusions.** Don't hard-code "don't flag inline imports" — instead say "don't flag intentional style choices."
4. **Pipeline entry points and MCP tools use `async def`.** Internal helpers may be sync where appropriate.
5. **Tests mock `ctx.sample()` and `GitHubPRClient`.** No real API calls in unit tests.
6. **Extend by subclassing.** Override prompt class attributes and/or step methods (e.g. `SecurityReview(ThoroughReview)`).
