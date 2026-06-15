"""End-to-end trace of the Debby debate agent through MLflow.

Bundles the local debby agent directory as a gzipped tarball, posts
it via ``OmnigentClient.sessions_chat(bundle=..., hooks=...)``, and
lets the new SessionsChat hook surface (omnigent PR #43, merged
2026-06-14) fire all the StreamHooks the adapter wraps. The result
is a complete MLflow trace under experiment ``omnigent-debby``.

Prereqs
-------

1. ``omnigent`` installed and ``omni setup`` run so a Claude and an
   OpenAI provider are configured. The debby example needs both.

2. ``mlflow>=3.0`` installed and pointed somewhere writable. By
   default the script uses a local SQLite store (set with
   ``MLFLOW_TRACKING_URI``). For Databricks, set
   ``MLFLOW_TRACKING_URI=databricks`` and
   ``MLFLOW_EXPERIMENT_NAME=/Users/<you>/omnigent-debby``.

3. ``omnigent-mlflow`` installed (``pip install -e ..`` from this
   examples folder, or ``pip install omnigent-mlflow``).

Run
---

::

    python trace_debby.py "what tier should we ship the free plan at?"

Then open ``mlflow ui`` and look under the omnigent-debby experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import pathlib
import sys
import tarfile

from omnigent_client import LocalServer, OmnigentClient

from omnigent_mlflow import OmnigentMlflowHooks

DEFAULT_QUESTION = "what is the right pricing tier for a developer plan?"


def _bundle_agent(agent_path: str) -> bytes:
    """Gzipped tar of the agent directory contents, suitable for sessions_chat.

    The server expects ``config.yaml`` at the root of the tarball, so
    add each entry inside the agent directory with ``arcname`` rooted
    at "." rather than under the directory name.
    """
    p = pathlib.Path(agent_path).resolve()
    if not p.is_dir():
        raise ValueError(f"Agent path is not a directory: {p}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for child in sorted(p.rglob("*")):
            rel = child.relative_to(p)
            tf.add(child, arcname=str(rel), recursive=False)
    return buf.getvalue()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--server",
        default=os.environ.get("OMNIGENT_SERVER"),
        help="Existing omnigent server URL. If unset, a local server is spawned.",
    )
    parser.add_argument(
        "--agent-path",
        default=os.environ.get("OMNIGENT_AGENT_PATH"),
        help=(
            "Path to the agent directory (e.g. examples/debby/ from the "
            "omnigent repo). Tar-gzipped and posted as the session bundle."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "omnigent-debby"),
    )
    args = parser.parse_args()

    if not args.agent_path:
        print(
            "ERROR: --agent-path (or OMNIGENT_AGENT_PATH env) is required. "
            "Point it at the agent directory, e.g. /path/to/omnigent/examples/debby.",
            file=sys.stderr,
        )
        return 2

    hooks_factory = OmnigentMlflowHooks(experiment=args.experiment)
    bundle = _bundle_agent(args.agent_path)
    print(f"Bundled {args.agent_path} -> {len(bundle)} bytes", file=sys.stderr)

    if args.server:
        async with OmnigentClient(base_url=args.server) as client:
            await _run_one(client, bundle, args.question, hooks_factory)
    else:
        async with LocalServer(agent_path=args.agent_path) as server:
            await _run_one(server.client, bundle, args.question, hooks_factory)
    return 0


async def _bind_runner(
    client: OmnigentClient, session_id: str, *, base_url: str | None = None
) -> str | None:
    """List the server's runners and PATCH the first online one onto the
    session so it has an executor before send().

    omnigent's CLI does this implicitly through ``omni run``; for a
    pure-SDK caller we have to do it explicitly. Returns the bound
    runner_id on success, None when no online runner is registered
    (the caller can still try send() and surface the resulting
    ``No runner bound for session`` error).
    """
    import httpx

    base = base_url or str(client._http.base_url).rstrip("/")  # type: ignore[attr-defined]
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    async with httpx.AsyncClient() as h:
        r = await h.get(f"{base}/v1/runners")
        r.raise_for_status()
        runners = r.json().get("data", [])
        online = [r for r in runners if r.get("online")]
        if not online:
            return None
        runner_id = online[0]["runner_id"]
        await h.patch(
            f"{base}/v1/sessions/{session_id}",
            json={"runner_id": runner_id},
        )
        return runner_id


async def _run_one(
    client: OmnigentClient,
    bundle: bytes,
    question: str,
    hooks_factory: OmnigentMlflowHooks,
) -> None:
    chat = await client.sessions_chat(bundle=bundle, hooks=hooks_factory.stream_hooks())
    session_id = getattr(getattr(chat, "_session", None), "id", "?")
    print(f"Session: {session_id}", file=sys.stderr)
    runner_id = await _bind_runner(client, session_id, base_url=os.environ.get("OMNIGENT_SERVER"))
    if runner_id:
        print(f"Bound runner: {runner_id}", file=sys.stderr)
    async for event in chat.send(question):
        # Hooks capture the structural signal; print text deltas for
        # the operator. Drop anything not display-worthy.
        text = _extract_text(event)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
    print()


def _extract_text(event: object) -> str | None:
    """Pull display text out of common event shapes."""
    delta = getattr(event, "delta", None)
    if isinstance(delta, str):
        return delta
    text_attr = getattr(event, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return None


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
