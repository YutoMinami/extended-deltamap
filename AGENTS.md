# AGENTS

## Purpose
- This file records repo-local working agreements for AI agents and other automated contributors.
- Follow these instructions together with any task-specific user request.

## Current priorities
- Work through the current refactor plan in `TODO.md`.
- Prioritize operational correctness first, then cleanup and documentation.

## Required work order
1. Silence current warnings.
2. Fix path handling.
3. Add type hints.
4. Normalize indentation and formatting.
5. Add Google-style docstrings.

## Commit cadence
- Make progress in small, reviewable steps.
- Create commits at appropriate step boundaries instead of batching unrelated changes together.
- At minimum, make a separate commit after each major step in the required work order when the tree is in a valid state.
- If a step becomes large, split it into multiple commits with clear scopes.

## Change scope
- Avoid mixing behavioral fixes, formatting-only changes, and documentation-only changes in the same commit unless the user explicitly asks for that.
- Prefer finishing one layer of work cleanly before starting the next.
- When touching legacy research code, preserve behavior unless the current step is explicitly about fixing behavior.

## Validation expectations
- Run the smallest meaningful validation after each step.
- Prefer `uv run` commands so checks use the managed environment.
- If validation cannot be completed, document what was attempted and what remains uncertain.

## Notes for this repository
- The canonical package code lives under `extended_deltamap/`.
- Example scripts under `examples/` are still operational entry points and may need careful path handling.
- Local large data can live under `data/`, which is gitignored.
