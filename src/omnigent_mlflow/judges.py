"""LLM-as-judge scorers for omnigent agent output.

These scorers are MLflow ``judge.align``-compatible: they accept a
``trace`` (an MLflow Trace object) and return one or more
``Feedback`` instances. Register them with ``mlflow.genai.scorers``
to evaluate every traced omnigent run automatically.

The two scorers shipped here are illustrative; both target the Debby
debate pattern. Adapt or replace them for your own agent shape.

* ``debate_synthesis_quality``. On a Debby trace, finds the two
  sub-agent (claude, gpt) outputs and the final synthesis, judges
  whether the synthesis fairly represents both sides without losing
  the substantive disagreement.
* ``tool_call_efficiency``. Judges whether the agent issued
  redundant tool calls (the same call with identical arguments).
  Useful for catching loops where a model retries the same query
  in slightly different phrasings.

Why client-side judging: omnigent's StreamHooks fire on the client,
so the trace already lives where MLflow can read it. We do not need
to instrument the server.
"""

from __future__ import annotations

from typing import Any


_DEBATE_SCORER_PROMPT = """\
You are auditing the output of a multi-agent debate.

Below are three texts: agent A's answer, agent B's answer, and the
final synthesis the orchestrator presented to the user.

Score the synthesis on three dimensions, each 1-5:

1. fair_representation. Does the synthesis fairly attribute the
   distinct positions to A and B, or does it silently merge them?
2. surfaces_disagreement. Does the synthesis name the real
   disagreement, or does it paper over it?
3. accuracy. Does the synthesis contain claims not supported by
   either A or B?

Return JSON: {{"fair_representation": int, "surfaces_disagreement":
int, "accuracy": int, "rationale": str}}.

Agent A (claude):
{a_output}

Agent B (gpt):
{b_output}

Synthesis:
{synthesis}
"""


def debate_synthesis_quality(trace: Any) -> list[Any]:
    """Judge whether a Debby trace's final synthesis fairly represents
    both sub-agents.

    Walks the trace's span tree, pulls the two sub_agent.* spans'
    outputs, finds the parent agent.* span's output as the synthesis,
    and asks a judge model to score the three dimensions above.

    Returns a list of ``mlflow.entities.Feedback``. Empty if the
    trace does not match the Debby shape (two sub_agent spans named
    ``claude`` and ``gpt``).
    """
    try:
        from mlflow.entities import Feedback
        from mlflow.genai.judges import make_judge
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "mlflow.genai is required for judges; pip install 'mlflow[genai]'"
        ) from e

    spans = list(trace.data.spans) if hasattr(trace, "data") else []
    a_out = _find_sub_agent_output(spans, "claude")
    b_out = _find_sub_agent_output(spans, "gpt")
    synthesis = _find_root_agent_output(spans)
    if not (a_out and b_out and synthesis):
        return []

    judge = make_judge(
        name="debate_synthesis_quality",
        instructions=_DEBATE_SCORER_PROMPT.format(
            a_output=a_out, b_output=b_out, synthesis=synthesis
        ),
        model="endpoints:/databricks-claude-sonnet-4-6",
    )
    verdict = judge(trace=trace)
    if isinstance(verdict, Feedback):
        return [verdict]
    return [verdict] if verdict else []


def tool_call_efficiency(trace: Any) -> list[Any]:
    """Flag a trace as inefficient when the same tool is called with
    identical arguments more than once in a single response.

    Pure code, no model call. Cheap to run on every trace.
    """
    try:
        from mlflow.entities import Feedback
    except ImportError as e:  # pragma: no cover
        raise ImportError("mlflow is required for judges") from e

    spans = list(trace.data.spans) if hasattr(trace, "data") else []
    seen: dict[tuple[str, str], int] = {}
    duplicates = 0
    for s in spans:
        attrs = getattr(s, "attributes", {}) or {}
        if attrs.get("omnigent.tool.name"):
            key = (
                str(attrs.get("omnigent.tool.name", "")),
                _stringify(getattr(s, "inputs", None)),
            )
            if key in seen:
                duplicates += 1
            seen[key] = seen.get(key, 0) + 1
    return [
        Feedback(
            name="tool_call_efficiency",
            value=("good" if duplicates == 0 else "wasted"),
            rationale=f"{duplicates} duplicate tool calls observed",
        )
    ]


def _find_sub_agent_output(spans: list[Any], agent_name: str) -> str | None:
    for s in spans:
        attrs = getattr(s, "attributes", {}) or {}
        if attrs.get("omnigent.sub_agent.name") == agent_name:
            outputs = getattr(s, "outputs", None)
            if isinstance(outputs, dict):
                return _stringify(outputs.get("summary"))
    return None


def _find_root_agent_output(spans: list[Any]) -> str | None:
    # The root agent span has no parent_id; pick the deepest output
    # text from it.
    for s in spans:
        if getattr(s, "parent_id", None):
            continue
        outputs = getattr(s, "outputs", None)
        if isinstance(outputs, dict):
            return _stringify(outputs.get("output"))
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
