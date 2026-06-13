"""Score a Debby debate trace with the bundled judges.

Run this after ``trace_debby.py`` has produced at least one trace in
the ``omnigent-debby`` experiment. The script picks the most recent
trace, runs both shipped scorers (debate_synthesis_quality and
tool_call_efficiency), and prints the verdicts.

Run::

    python examples/debate_judge.py

Or point at a specific trace::

    python examples/debate_judge.py --trace-id <trace_id>
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow

from omnigent_mlflow.judges import debate_synthesis_quality, tool_call_efficiency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "omnigent-debby")
    )
    parser.add_argument("--trace-id", default=None)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    client = mlflow.MlflowClient()

    if args.trace_id:
        trace = client.get_trace(args.trace_id)
    else:
        traces = mlflow.search_traces(
            experiment_names=[args.experiment],
            max_results=1,
            order_by=["timestamp DESC"],
        )
        if not len(traces):
            print(f"No traces found in experiment {args.experiment!r}.", file=sys.stderr)
            print("Run examples/trace_debby.py first.", file=sys.stderr)
            return 1
        trace_id = traces.iloc[0]["trace_id"]
        trace = client.get_trace(trace_id)

    print(f"Scoring trace {trace.info.trace_id}")

    eff = tool_call_efficiency(trace)
    for fb in eff:
        print(f"  {fb.name:30s} value={fb.value!r}  ({fb.rationale})")

    debate = debate_synthesis_quality(trace)
    for fb in debate:
        print(f"  {fb.name:30s} value={fb.value!r}")
        if fb.rationale:
            print(f"    rationale: {fb.rationale[:200]}")

    if not eff and not debate:
        print("  no scorers matched the trace shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
