"""MLflow tracing adapter for omnigent's StreamHooks.

The adapter holds a stack of active spans and maps each omnigent
lifecycle callback to one MLflow span boundary. Span types are picked
to match MLflow's GenAI conventions (AGENT for response/sub-agent
roots, TOOL for tool calls, LLM for messages, CHAIN for reasoning
and compaction).

Spans are created with mlflow.start_span_no_context() so the manual
start/end pairing matches omnigent's callback shape. A dictionary
keyed by omnigent identifiers (response_id, call_id, sub-agent id)
maps each start callback to its open span so the corresponding end
callback can close it cleanly even when callbacks arrive
interleaved.

Why not use MLflow autolog: autolog instruments the LLM SDK calls
the omnigent server makes server-side, which the client never sees.
The client-visible signal is StreamHooks, so that is where the
client-side trace originates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanType

# omnigent_client is an optional runtime dep. Hook method signatures
# use Any so the type checker does not need omnigent_client installed
# to validate this module. At runtime stream_hooks() resolves the
# real StreamHooks class and fails loudly if it isn't installed.
if TYPE_CHECKING:  # pragma: no cover
    from omnigent_client import StreamHooks  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class _SpanRecord:
    """Internal handle for an open MLflow span."""

    span: Any
    name: str
    started_at: float


@dataclass
class OmnigentMlflowHooks:
    """MLflow tracing adapter for omnigent.

    Construct one per session (sessions are independent traces).
    Attach to a stream with ``stream_hooks()``.

    Args:
        experiment: MLflow experiment name. Created if missing.
        tracking_uri: Optional override for MLflow tracking. Defaults
            to whatever ``mlflow.get_tracking_uri()`` returns, which
            picks up ``MLFLOW_TRACKING_URI`` env or the Databricks
            workspace context inside a notebook.
        capture_inputs: When True the adapter copies tool arguments
            and user message content into the span as ``inputs``.
            Disable for PII-sensitive workloads.
        capture_outputs: When True the adapter copies tool results
            and assistant content into the span as ``outputs``.
            Disable for PII-sensitive workloads.
    """

    experiment: str = "omnigent"
    tracking_uri: str | None = None
    capture_inputs: bool = True
    capture_outputs: bool = True

    # Span registry: response_id / call_id / sub-agent id -> SpanRecord.
    # Public attribute access is intentional so callers can inspect
    # what's currently open for debugging without reaching into
    # private state.
    open_spans: dict[str, _SpanRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

    # --------------------------------------------------------------
    # Public surface
    # --------------------------------------------------------------

    def stream_hooks(self) -> StreamHooks:
        """Return a ``StreamHooks`` instance bound to this adapter.

        Pass the result into ``BlockStream(hooks=...)`` or anywhere
        else omnigent accepts a ``StreamHooks`` value.
        """
        try:
            from omnigent_client import StreamHooks
        except ImportError as e:
            raise ImportError(
                "omnigent_client is not installed. Run: pip install omnigent-client"
            ) from e
        return StreamHooks(
            on_response_start=self._on_response_start,
            on_response_end=self._on_response_end,
            on_tool_call_start=self._on_tool_call_start,
            on_tool_call_end=self._on_tool_call_end,
            on_message_start=self._on_message_start,
            on_message_end=self._on_message_end,
            on_reasoning_start=self._on_reasoning_start,
            on_reasoning_end=self._on_reasoning_end,
            on_compaction_start=self._on_compaction_start,
            on_compaction_end=self._on_compaction_end,
            on_sub_agent_spawned=self._on_sub_agent_spawned,
            on_sub_agent_completed=self._on_sub_agent_completed,
            on_retry=self._on_retry,
            on_server_error=self._on_server_error,
        )

    # --------------------------------------------------------------
    # Response lifecycle (root span per agent turn)
    # --------------------------------------------------------------

    def _on_response_start(self, ctx: Any) -> None:
        r = ctx.response
        span = mlflow.start_span_no_context(
            name=f"agent.{_safe(r.model)}",
            span_type=SpanType.AGENT,
        )
        span.set_attribute("omnigent.response_id", r.id)
        span.set_attribute("omnigent.model", r.model)
        if r.previous_response_id:
            span.set_attribute("omnigent.previous_response_id", r.previous_response_id)
        if r.conversation:
            span.set_attribute("omnigent.conversation_id", r.conversation.id)
        if self.capture_inputs and r.instructions:
            span.set_inputs({"instructions_preview": r.instructions[:1000]})
        self.open_spans[r.id] = _SpanRecord(span=span, name=span.name, started_at=r.created_at)

    def _on_response_end(self, ctx: Any) -> None:
        r = ctx.response
        rec = self.open_spans.pop(r.id, None)
        if rec is None:
            logger.debug("response_end for unknown id %s", r.id)
            return
        rec.span.set_attribute("omnigent.status", ctx.status)
        if r.usage:
            rec.span.set_attribute("omnigent.tokens.input", r.usage.input_tokens)
            rec.span.set_attribute("omnigent.tokens.output", r.usage.output_tokens)
            rec.span.set_attribute("omnigent.tokens.total", r.usage.total_tokens)
        if r.error:
            rec.span.set_attribute("omnigent.error.code", r.error.code)
            rec.span.set_attribute("omnigent.error.message", r.error.message)
            rec.span.set_status("ERROR")
        if self.capture_outputs:
            rec.span.set_outputs({"output": r.output})
        rec.span.end()

    # --------------------------------------------------------------
    # Tool calls
    # --------------------------------------------------------------

    def _on_tool_call_start(self, ctx: Any) -> None:
        span = mlflow.start_span_no_context(
            name=f"tool.{ctx.name}",
            span_type=SpanType.TOOL,
        )
        span.set_attribute("omnigent.tool.name", ctx.name)
        span.set_attribute("omnigent.tool.call_id", ctx.call_id)
        span.set_attribute("omnigent.agent_name", ctx.agent_name)
        span.set_attribute("omnigent.tool.executed_by", ctx.executed_by)
        if self.capture_inputs:
            span.set_inputs({"arguments": ctx.arguments})
        self.open_spans[ctx.call_id] = _SpanRecord(span=span, name=span.name, started_at=0.0)

    def _on_tool_call_end(self, ctx: Any) -> None:
        rec = self.open_spans.pop(ctx.call_id, None)
        if rec is None:
            logger.debug("tool_call_end for unknown id %s", ctx.call_id)
            return
        if self.capture_outputs:
            rec.span.set_outputs({"output": ctx.output})
        rec.span.end()

    # --------------------------------------------------------------
    # Messages (LLM-level)
    # --------------------------------------------------------------

    def _on_message_start(self, ctx: Any) -> None:
        span = mlflow.start_span_no_context(name="llm.message", span_type=SpanType.LLM)
        span.set_attribute("omnigent.response_id", ctx.response_id)
        self.open_spans[f"msg:{ctx.response_id}"] = _SpanRecord(
            span=span, name=span.name, started_at=0.0
        )

    def _on_message_end(self, ctx: Any) -> None:
        # MessageEndCtx has no response_id, so we close the most
        # recently opened message span. omnigent emits message events
        # sequentially per response, so the last-opened heuristic is
        # correct in practice; if a future server emits them in
        # parallel, this is the seam to revisit.
        key = next(
            (k for k in reversed(list(self.open_spans)) if k.startswith("msg:")),
            None,
        )
        if not key:
            logger.debug("message_end with no open message span")
            return
        rec = self.open_spans.pop(key)
        if self.capture_outputs:
            rec.span.set_outputs({"content": ctx.content})
        rec.span.end()

    # --------------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------------

    def _on_reasoning_start(self, ctx: Any) -> None:
        span = mlflow.start_span_no_context(name="reasoning", span_type=SpanType.CHAIN)
        self.open_spans["reasoning"] = _SpanRecord(span=span, name=span.name, started_at=0.0)

    def _on_reasoning_end(self, ctx: Any) -> None:
        rec = self.open_spans.pop("reasoning", None)
        if rec is None:
            return
        if self.capture_outputs:
            rec.span.set_outputs({"reasoning": ctx.reasoning_text, "summary": ctx.summary_text})
        rec.span.end()

    # --------------------------------------------------------------
    # Compaction
    # --------------------------------------------------------------

    def _on_compaction_start(self, ctx: Any) -> None:
        span = mlflow.start_span_no_context(name="compaction", span_type=SpanType.CHAIN)
        self.open_spans["compaction"] = _SpanRecord(span=span, name=span.name, started_at=0.0)

    def _on_compaction_end(self, ctx: Any) -> None:
        rec = self.open_spans.pop("compaction", None)
        if rec is None:
            return
        if self.capture_outputs:
            rec.span.set_outputs({"item": ctx.item})
        rec.span.end()

    # --------------------------------------------------------------
    # Sub-agent delegation
    # --------------------------------------------------------------

    def _on_sub_agent_spawned(self, ctx: Any) -> None:
        for sub in ctx.sub_agents:
            span = mlflow.start_span_no_context(
                name=f"sub_agent.{sub.agent_name}",
                span_type=SpanType.AGENT,
            )
            span.set_attribute("omnigent.sub_agent.name", sub.agent_name)
            span.set_attribute("omnigent.sub_agent.response_id", sub.response_id)
            span.set_attribute("omnigent.parent_response_id", ctx.parent_response_id)
            self.open_spans[f"sub:{sub.response_id}"] = _SpanRecord(
                span=span, name=span.name, started_at=0.0
            )

    def _on_sub_agent_completed(self, ctx: Any) -> None:
        rec = self.open_spans.pop(f"sub:{ctx.response_id}", None)
        if rec is None:
            logger.debug("sub_agent_completed for unknown id %s", ctx.response_id)
            return
        rec.span.set_attribute("omnigent.sub_agent.status", ctx.status)
        if self.capture_outputs and ctx.output_summary:
            rec.span.set_outputs({"summary": ctx.output_summary})
        if ctx.status not in ("completed", "ok"):
            rec.span.set_status("ERROR")
        rec.span.end()

    # --------------------------------------------------------------
    # Error / retry annotations
    # --------------------------------------------------------------

    def _on_retry(self, ctx: Any) -> None:
        # Annotate every currently open span with the retry signal.
        # Retries are not their own spans because they are not
        # bounded events from the client's perspective; they are
        # marker annotations on whatever is currently running.
        for rec in self.open_spans.values():
            rec.span.set_attribute(f"omnigent.retry.{ctx.source}.attempt", ctx.attempt)
            rec.span.set_attribute(f"omnigent.retry.{ctx.source}.max_attempts", ctx.max_attempts)

    def _on_server_error(self, ctx: Any) -> None:
        for rec in self.open_spans.values():
            rec.span.set_attribute("omnigent.error.source", ctx.source)
            rec.span.set_attribute("omnigent.error.code", ctx.error.code)
            rec.span.set_attribute("omnigent.error.message", ctx.error.message)
            rec.span.set_status("ERROR")


def _safe(name: str) -> str:
    """MLflow span names allow most characters but stripping the few
    that confuse the UI search helps when filtering by agent."""
    return name.replace("/", "_").replace(" ", "_")
