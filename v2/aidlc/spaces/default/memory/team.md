# Team practices (FEDzk)

## Way of Working

- Dual mission always: research-publishable experiments + production-grade OSS.
- Docs must match public API in the same PR (CONCEPTS P9).
- Use docs/agent-kit HTML as the Cursor harness surface; AIDLC phases/stages structure the work.

## Code Style

- Prefer deep imports only until stable SDK exports exist; then export deliberately.
- Never silent-truncate model updates without Chunk Protocol v1.

## Security

- Untrack secrets/; gitleaks in pre-commit.
- Dual coordinator apps are forbidden — single verify-before-aggregate surface.
