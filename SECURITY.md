# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting instead:
**Security → Report a vulnerability** on this repository. You can expect an acknowledgement within
5 working days; if the report is confirmed, a fix or mitigation plan follows within 30 days.

Please include the affected version or commit, steps to reproduce and the impact you observed.

## Security-relevant design

See [docs/threat-model.md](docs/threat-model.md) for assets, threats and mitigations. In short:

| Area | Default |
|---|---|
| Prompt injection | Retrieved text is fenced in `<belgeler>` and the system prompt says it is data, not instructions; the model has no tools, so an injected instruction can at worst change the answer text, never trigger an action |
| Grounding | Citation markers outside the retrieved contexts are dropped; the API returns the cited chunks so a reader can check every claim; an abstention sentence is the required answer when the context lacks one |
| Database access | The API connects as `ragtr_app` (DML on the RAG tables only, no DDL); the schema is created by a one-shot `migrate` process as the owner role |
| SQL | Parameterised queries only (psycopg); the only formatted DDL value is the integer vector dimension, validated to 1..16000 |
| Input limits | Document ids match `^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$`; document size (500 000 chars) and question length (1 000 chars) are capped; `k` is bounded |
| Logging | Request bodies, questions and answers are not logged; logs carry ids, counts and timings |
| Container | `python:3.13-slim` runtime, non-root UID 10001, no build tools in the final stage |
| Local LLM | Generation runs on a local Ollama instance by default; no document text leaves the host |
| Secrets | None in the repository; compose defaults are for local development and can be overridden via `.env` |
| Supply chain | `uv.lock` pins every package (CPU-only PyTorch from the PyTorch index), updated by hand; CI runs `pip-audit --strict` over all extras and groups; CodeQL on every push |

**Not included:** authentication and authorisation, per-tenant document isolation, rate limiting, TLS
termination. This repository is a demonstration with synthetic data and must not be exposed as-is.
