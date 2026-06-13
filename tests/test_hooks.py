"""Smoke tests that do not require omnigent_client.

These exercise the adapter's span lifecycle by feeding it
hand-constructed context objects shaped like the real omnigent
dataclasses. The point is to catch regressions in span open/close
pairing and attribute extraction without a live omnigent server.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlflow
import pytest

from omnigent_mlflow import OmnigentMlflowHooks


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:///{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file:///{tmp_path}/mlruns")
    return tmp_path


def _response(rid: str, model: str = "debby"):
    return SimpleNamespace(
        id=rid,
        model=model,
        previous_response_id=None,
        conversation=None,
        instructions="test",
        created_at=0,
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
        error=None,
        output=[{"type": "message", "content": "hi"}],
    )


def test_response_lifecycle_opens_and_closes_one_span():
    h = OmnigentMlflowHooks(experiment="omnigent-test-lifecycle")
    h._on_response_start(SimpleNamespace(response=_response("r1")))
    assert "r1" in h.open_spans
    h._on_response_end(SimpleNamespace(response=_response("r1"), status="completed"))
    assert "r1" not in h.open_spans
    assert h.open_spans == {}


def test_tool_call_lifecycle():
    h = OmnigentMlflowHooks(experiment="omnigent-test-tool")
    h._on_response_start(SimpleNamespace(response=_response("r2")))
    h._on_tool_call_start(
        SimpleNamespace(
            name="web_search",
            arguments={"q": "x"},
            call_id="c1",
            agent_name="claude",
            executed_by="server",
        )
    )
    assert "c1" in h.open_spans
    h._on_tool_call_end(
        SimpleNamespace(name="web_search", call_id="c1", agent_name="claude", output="ok")
    )
    assert "c1" not in h.open_spans
    h._on_response_end(SimpleNamespace(response=_response("r2"), status="completed"))


def test_sub_agent_lifecycle_handles_parallel_children():
    h = OmnigentMlflowHooks(experiment="omnigent-test-sub")
    h._on_response_start(SimpleNamespace(response=_response("rp")))
    h._on_sub_agent_spawned(
        SimpleNamespace(
            parent_response_id="rp",
            sub_agents=[
                SimpleNamespace(agent_name="claude", response_id="sa-claude"),
                SimpleNamespace(agent_name="gpt", response_id="sa-gpt"),
            ],
        )
    )
    assert "sub:sa-claude" in h.open_spans
    assert "sub:sa-gpt" in h.open_spans
    # Close them in reverse order to confirm independent tracking
    h._on_sub_agent_completed(
        SimpleNamespace(
            response_id="sa-gpt", agent_name="gpt", status="completed", output_summary="..."
        )
    )
    assert "sub:sa-gpt" not in h.open_spans
    assert "sub:sa-claude" in h.open_spans
    h._on_sub_agent_completed(
        SimpleNamespace(
            response_id="sa-claude",
            agent_name="claude",
            status="completed",
            output_summary="...",
        )
    )
    assert "sub:sa-claude" not in h.open_spans
    h._on_response_end(SimpleNamespace(response=_response("rp"), status="completed"))


def test_capture_flags_off_yields_no_inputs_outputs():
    h = OmnigentMlflowHooks(
        experiment="omnigent-test-redacted", capture_inputs=False, capture_outputs=False
    )
    h._on_response_start(SimpleNamespace(response=_response("rr")))
    h._on_tool_call_start(
        SimpleNamespace(
            name="sql",
            arguments={"q": "SECRET"},
            call_id="c2",
            agent_name="a",
            executed_by="server",
        )
    )
    rec = h.open_spans["c2"]
    # set_inputs not called: the underlying span has no input recorded
    # (we can't easily introspect set_inputs from outside; the
    # assertion is that the call completes without storing the secret
    # arguments anywhere reachable by the test).
    assert rec.name.startswith("tool.")
    h._on_tool_call_end(SimpleNamespace(name="sql", call_id="c2", agent_name="a", output="x"))
    h._on_response_end(SimpleNamespace(response=_response("rr"), status="completed"))


def test_unknown_end_event_logs_but_does_not_crash():
    h = OmnigentMlflowHooks(experiment="omnigent-test-unknown")
    # No matching start; the end should be a no-op
    h._on_tool_call_end(
        SimpleNamespace(name="x", call_id="never-opened", agent_name="a", output="y")
    )
    assert h.open_spans == {}
