# omnigent-mlflow

Drop-in MLflow tracing for [omnigent](https://github.com/omnigent-ai/omnigent)
agents. One line to attach, full span tree per agent turn, sub-agent
delegation, tool calls, reasoning, and compaction all visible in the
MLflow Traces UI.

```python
from omnigent_client import BlockStream, OmnigentClient
from omnigent_mlflow import OmnigentMlflowHooks

hooks = OmnigentMlflowHooks(experiment="omnigent-debby")

async with OmnigentClient(base_url="http://localhost:8080") as client:
    session = client.session(model="debby")
    stream = BlockStream(hooks=hooks.stream_hooks())
    async for block in stream.stream(session, "design a pricing tier"):
        ...
```

That's it. Open `mlflow ui`, find the `omnigent-debby` experiment, and
the trace shows you:

```
agent.debby
├── llm.message               (debby's own reasoning)
├── sub_agent.claude
│   ├── llm.message
│   └── reasoning
├── sub_agent.gpt
│   ├── llm.message
│   └── tool.web_search
└── llm.message               (debby's final synthesis)
```

with input tokens, output tokens, total tokens, status per span, and
arguments + outputs as structured inputs/outputs.

## Why this exists

omnigent ships a clean `StreamHooks` callback surface in its Python
client SDK. Every lifecycle event you would want as a span has a
start/end hook pair: `on_response_start`/`on_response_end`,
`on_tool_call_start`/`on_tool_call_end`, `on_sub_agent_spawned`/
`on_sub_agent_completed`, plus reasoning, message, compaction, retry,
and error hooks.

MLflow ships first-class span-based GenAI tracing with the right
typed spans (AGENT, TOOL, LLM, CHAIN) and a UI that already knows how
to render them.

The two halves match one-to-one. This package is the 300-line adapter
that wires them together so you can stop staring at debug logs and
start filtering, comparing, and judging your agent runs in the
MLflow UI.

## What gets traced

| omnigent hook | MLflow span | type | notable attributes |
|---|---|---|---|
| `on_response_start/end` | `agent.<model>` | `AGENT` | response_id, model, conversation_id, status, token usage |
| `on_tool_call_start/end` | `tool.<name>` | `TOOL` | tool name, call_id, agent_name, executed_by, arguments, output |
| `on_message_start/end` | `llm.message` | `LLM` | response_id, content |
| `on_reasoning_start/end` | `reasoning` | `CHAIN` | reasoning text, summary |
| `on_compaction_start/end` | `compaction` | `CHAIN` | compaction item |
| `on_sub_agent_spawned/completed` | `sub_agent.<name>` | `AGENT` | sub-agent name, response_id, parent_response_id, status |
| `on_retry`, `on_server_error` | (annotation on parent) | n/a | retry attempt, error code/message, span status set to ERROR |

PII-sensitive workloads can disable input/output capture:

```python
OmnigentMlflowHooks(capture_inputs=False, capture_outputs=False)
```

Span attributes still record the structural signal (which tool ran
how many times, retries, errors, token counts) so a redacted trace
is still useful for debugging.

## Install

```bash
pip install omnigent-mlflow
```

You also need `omnigent-client` (omnigent's Python SDK) and `mlflow`:

```bash
pip install omnigent-client "mlflow>=3.0"
```

## Where to write traces

The adapter calls `mlflow.set_experiment(name)` and otherwise uses
whatever tracking URI is already configured. Three common setups:

* Local file store: leave `MLFLOW_TRACKING_URI` unset. Traces land in
  `./mlruns`. `mlflow ui` to browse.
* Self-hosted MLflow: `export MLFLOW_TRACKING_URI=http://your-mlflow:5000`.
* Databricks: `export MLFLOW_TRACKING_URI=databricks` and
  `MLFLOW_EXPERIMENT_NAME=/Users/<you>/omnigent-debby`. Traces appear in
  the workspace UI under MLflow Experiments and on the `traces` tab of
  the experiment.

## Scoring traces

`omnigent_mlflow.judges` ships two illustrative scorers:

* `debate_synthesis_quality` looks at Debby's trace, pulls the two
  sub-agent outputs and the orchestrator's final synthesis, and
  judges whether the synthesis fairly represents both sides.
* `tool_call_efficiency` walks the span tree and flags duplicate tool
  calls (same name + same arguments).

Register them with `mlflow.genai.scorers` to run on every trace:

```python
import mlflow
from omnigent_mlflow.judges import debate_synthesis_quality, tool_call_efficiency

mlflow.genai.scorers.register(debate_synthesis_quality)
mlflow.genai.scorers.register(tool_call_efficiency)
```

For prompt iteration, point `mlflow.genai.judge.align` at any of
these scorers and a labeled dataset; the optimizer refines the
judge's instructions against your ground truth.

## Example: trace Debby end to end

`examples/trace_debby.py` spins up an omnigent local server, attaches
the adapter, runs Debby on one question, and exits. Run it with:

```bash
python examples/trace_debby.py "what tier should we ship the free plan at?"
mlflow ui  # open http://localhost:5000
```

The script also accepts `--server` to use an existing omnigent server
and `--agent` to point at any agent other than Debby.

## Design notes

A few choices worth being explicit about:

* **Span identifiers come from omnigent, not from MLflow.** Each
  `agent.*` span carries the `response_id` as an attribute, each
  `tool.*` span carries the `call_id`. Sub-agent spans carry both.
  This keeps client-side spans correlatable with server-side logs.
* **Start and end are decoupled.** omnigent emits `*_start` and
  `*_end` callbacks separately. The adapter holds a dict from
  omnigent's id (response_id / call_id / sub-agent id) to the open
  span and closes on the matching `*_end`. Cleanly handles
  interleaved tool calls and parallel sub-agents.
* **Retries and errors annotate the parent span, not their own span.**
  A retry is not a bounded event from the client's perspective. It's
  a marker on whatever was running. Same for `on_server_error`.
* **No autolog conflict.** This adapter does not call `mlflow.autolog`
  on any underlying LLM provider. If you separately enable autolog
  on the same process, you get both: server-side LLM call spans
  (from autolog instrumentation of the provider SDK) AND client-side
  omnigent lifecycle spans (from this adapter). They land in the
  same trace via MLflow's active span context.

## Status

Alpha. Tracks omnigent v0.1.x. Live end to end against the bundled
debby example works as of 2026-06-15.

### Worked end to end

omnigent PR [#43](https://github.com/omnigent-ai/omnigent/pull/43)
("Expose sessions chat stream hooks") by
[@dipeshbabu](https://github.com/dipeshbabu), reviewed and merged by
[@dbczumar](https://github.com/dbczumar), wired `StreamHooks` into
the sessions-first API. That unblocked this adapter end to end. The
follow-up patch [#149](https://github.com/omnigent-ai/omnigent/pull/149)
fixed a circular import in `omnigent.llms` that was blocking server
startup on a fresh install.

A real trace from a debby session:

![MLflow trace list with 6 spans from a real debby debate session](docs/screenshots/mlflow-traces-list.png)

Trace detail showing the orchestrator's "both partners working on it
in parallel" output:

![Trace detail view with response inputs/outputs](docs/screenshots/mlflow-trace-detail.png)

### Known issues still open upstream

- `StreamHooks.on_sub_agent_spawned` and `on_sub_agent_completed`
  are declared but never fired anywhere in the SDK ([omnigent#146](https://github.com/omnigent-ai/omnigent/issues/146)).
  Sub-agent spans don't appear in the trace tree today; they will
  once the maintainers wire those callbacks.
- Pure-SDK examples need to PATCH a `runner_id` onto a freshly-
  created session before `send()` works. `examples/trace_debby.py`
  does this explicitly. The omnigent CLI does it implicitly; the
  SDK doesn't have a helper yet.

## Credits

The omnigent team at Databricks for designing `StreamHooks` so it
maps cleanly onto a tracing model. The MLflow team for span types
that already speak the GenAI vocabulary. The MLflow community for
`judge.align` and the scorer registry.

If you find a bug or want a new hook surfaced, open an issue or PR.

## License

Apache 2.0. See `LICENSE`.
