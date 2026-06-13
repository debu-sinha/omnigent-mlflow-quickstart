"""End-to-end trace of the Debby debate agent.

Spins up an omnigent server (or connects to a running one), attaches
OmnigentMlflowHooks, runs Debby on a single question, and exits when
the response completes. The resulting trace is in MLflow under
experiment ``omnigent-debby``.

Prereqs
-------

1. ``omnigent`` installed and ``omni setup`` run so a Claude and an
   OpenAI provider are configured. The debby example needs both
   harnesses.

2. ``mlflow`` installed and pointed at where you want to write
   traces. The default sets an experiment on the local store
   (``./mlruns``); for Databricks set ``MLFLOW_TRACKING_URI=databricks``
   and ``MLFLOW_EXPERIMENT_NAME=/Users/<you>/omnigent-debby`` before
   running this script.

3. ``omnigent-mlflow`` installed (``pip install -e ..`` from this
   examples folder, or ``pip install omnigent-mlflow`` once
   published).

Run
---

::

    python trace_debby.py "what tier should we ship the free plan at?"

Then open ``mlflow ui`` and look under the omnigent-debby experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from omnigent_client import BlockStream, LocalServer, OmnigentClient

from omnigent_mlflow import OmnigentMlflowHooks


DEFAULT_QUESTION = "what is the right pricing tier for a developer plan?"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--server",
        default=os.environ.get("OMNIGENT_SERVER"),
        help="Existing omnigent server URL. If unset, a local server is spawned.",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("OMNIGENT_AGENT", "debby"),
        help="Agent name to send the question to.",
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "omnigent-debby"),
    )
    args = parser.parse_args()

    hooks_factory = OmnigentMlflowHooks(experiment=args.experiment)

    server_ctx = LocalServer() if not args.server else _noop_ctx()
    async with server_ctx as srv:
        base_url = args.server or srv.url
        async with OmnigentClient(base_url=base_url) as client:
            session = client.session(model=args.agent)
            stream = BlockStream(hooks=hooks_factory.stream_hooks())
            async for block in stream.stream(session, args.question):
                # The hook-based tracer captures everything we need;
                # the loop here just prints assistant text to stdout
                # for the operator. Filter out the chunks you don't
                # want printed (reasoning, tool args, etc.) per taste.
                text = _extract_text(block)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
        print()  # newline after streaming completes
    return 0


def _extract_text(block: object) -> str | None:
    """Pull display text out of common block shapes, return None for
    blocks the operator probably does not want streamed verbatim."""
    text_attr = getattr(block, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    chunk = getattr(block, "chunk", None)
    if isinstance(chunk, str):
        return chunk
    return None


class _noop_ctx:
    """Async context manager that does nothing. Used when the caller
    already brought their own omnigent server."""

    async def __aenter__(self) -> "_noop_ctx":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def url(self) -> str:  # pragma: no cover - never called
        raise RuntimeError("_noop_ctx has no URL; pass --server")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
