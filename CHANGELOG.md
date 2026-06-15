# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/spec/v2.0.0.html) once
the project leaves alpha. While in alpha, breaking changes can land on
`main` and they're called out here under `### Changed (breaking)`.

## [Unreleased]

### Added
- `OmnigentMlflowHooks` adapter mapping every `StreamHooks` callback
  to an MLflow span (AGENT / TOOL / LLM / CHAIN), nested under one
  trace per agent turn via a response stack and `parent_span=` wiring.
- `omnigent_mlflow.judges` with two illustrative scorers:
  `debate_synthesis_quality` (LLM-as-judge for the Debby debate
  pattern) and `tool_call_efficiency` (code-only duplicate-call
  detection).
- `examples/trace_debby.py` end-to-end against the bundled debby
  agent; bundles the agent directory, PATCHes a runner onto the new
  session, sends a turn, captures spans.
- `examples/debate_judge.py` runs the bundled scorers against a
  produced trace.
- Architecture diagram and repo logo under `docs/`.
- Five unit tests covering the response stack, parent-child wiring,
  parallel sub-agent handling, PII redaction flags, and unknown-end
  events.

### Fixed
- Span hierarchy: spans now nest under one trace per agent turn,
  not one trace per span. Open response stack tracks the current
  parent on every `_on_*_start` callback.

### Notes
- This release line tracks omnigent v0.1.x. It depends on omnigent
  PRs [#43](https://github.com/omnigent-ai/omnigent/pull/43) (sessions-
  first hooks) and [#149](https://github.com/omnigent-ai/omnigent/pull/149)
  (server startup fix). Without those merged, the example here won't
  reach the SSE stream cleanly.
