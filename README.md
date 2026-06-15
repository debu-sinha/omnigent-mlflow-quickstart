# omnigent-mlflow

MLflow tracing for [omnigent](https://github.com/omnigent-ai/omnigent)
sessions. Attach `OmnigentMlflowHooks` to a session; every `StreamHooks`
callback becomes an MLflow span.

```python
import mlflow
from omnigent_client import OmnigentClient
from omnigent_mlflow import OmnigentMlflowHooks

mlflow.set_experiment("my-agent")

hooks = OmnigentMlflowHooks()
async with OmnigentClient(base_url="http://localhost:6767") as client:
    chat = await client.sessions_chat(
        bundle=open("agent.tar.gz", "rb").read(),
        hooks=hooks.stream_hooks(),
    )
    async for event in chat.send("hello"):
        ...
```

That's the integration. Open `mlflow ui`, find the experiment, click any
trace.

## Install

```bash
pip install omnigent-mlflow
```

Requires `omnigent>=0.1` and `mlflow>=3.0`. They're declared in
`pyproject.toml`, so a fresh `pip install omnigent-mlflow` pulls
them in.

## Span mapping

Each omnigent `StreamHooks` callback maps to one MLflow span:

| omnigent hook | MLflow span name | span type |
| --- | --- | --- |
| `on_response_start` / `on_response_end` | `agent.<model>` | `AGENT` |
| `on_tool_call_start` / `on_tool_call_end` | `tool.<name>` | `TOOL` |
| `on_message_start` / `on_message_end` | `llm.message` | `LLM` |
| `on_reasoning_start` / `on_reasoning_end` | `reasoning` | `CHAIN` |
| `on_compaction_start` / `on_compaction_end` | `compaction` | `CHAIN` |
| `on_sub_agent_spawned` / `on_sub_agent_completed` | `sub_agent.<name>` | `AGENT` |
| `on_retry`, `on_server_error` | annotation on the open span | n/a |

Tool spans carry their `arguments` as inputs and `output` as outputs.
Response and sub-agent spans carry token usage on the matching `_end`.
Retries annotate every currently-open span with attempt counts.

For PII-sensitive workloads, disable payload capture and you'll keep
only the structural signal:

```python
OmnigentMlflowHooks(capture_inputs=False, capture_outputs=False)
```

## Worked example

`examples/trace_debby.py` runs the bundled debby agent end to end
against an omnigent server you already have running, and emits a real
MLflow trace.

```bash
# In one terminal: an omnigent server with the debby agent loaded
omni debby

# In another terminal: stream a turn through the adapter
export OMNIGENT_SERVER=http://localhost:6767
export OMNIGENT_AGENT_PATH=/path/to/omnigent/examples/debby
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python examples/trace_debby.py "design a pricing tier"

# Then open the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

A real run produces six spans for a single Debby turn: the
orchestrator's first response, two `sys_session_send` tool calls
dispatching to the Claude and GPT sub-agents, the reasoning span,
the closing message, and a status span.

![MLflow trace list, six spans from a real debby session](docs/screenshots/mlflow-traces-list.png)

The omnigent server it ran against:

![omnigent server OpenAPI page](docs/screenshots/omnigent-server-openapi.png)

## Scoring traces

`omnigent_mlflow.judges` ships two examples:

* `tool_call_efficiency` walks the span tree and flags duplicate
  tool calls (same name + same arguments). Code-only, cheap to run on
  every trace.
* `debate_synthesis_quality` looks at a Debby trace, pulls the two
  sub-agent outputs and the orchestrator's synthesis, and asks a
  judge model whether the synthesis fairly attributes positions and
  surfaces real disagreement.

Wire them up with `mlflow.genai.scorers` to run on every trace.

## Status and open issues

Alpha. The adapter pairs with these upstream changes:

* omnigent [#43](https://github.com/omnigent-ai/omnigent/pull/43)
  added `StreamHooks` support to `SessionsChat`. Without it, hooks
  do not fire on the sessions-first API.
* omnigent [#149](https://github.com/omnigent-ai/omnigent/pull/149)
  fixed a circular import that blocked server startup on a fresh
  install of main.

Still open:

* omnigent [#146](https://github.com/omnigent-ai/omnigent/issues/146)
  observes that `StreamHooks.on_sub_agent_spawned` /
  `on_sub_agent_completed` are declared but never fired. Sub-agent
  spans won't appear in the trace tree until those callbacks are
  wired.
* The pure-SDK example has to PATCH a `runner_id` onto a freshly-
  created session before `send()` works. The omnigent CLI does this
  implicitly via `omni run`. The SDK doesn't have a helper for it
  yet, so `examples/trace_debby.py` does it explicitly.

## Tests

```bash
pytest -q
```

Five unit tests cover the lifecycle pairing, attribute extraction,
parallel sub-agent handling, PII redaction flags, and unknown-end-
event handling. They use `SimpleNamespace`-shaped context objects so
they run without a live omnigent server.

## License

Apache 2.0. See `LICENSE`.
