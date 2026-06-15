"""MLflow tracing adapter for omnigent agents.

Usage::

    import mlflow
    from omnigent_client import OmnigentClient
    from omnigent_mlflow import OmnigentMlflowHooks

    mlflow.set_experiment("omnigent-debby")
    hooks = OmnigentMlflowHooks()

    async with OmnigentClient(base_url="http://localhost:6767") as client:
        chat = await client.sessions_chat(
            bundle=open("agent.tar.gz", "rb").read(),
            hooks=hooks.stream_hooks(),
        )
        async for event in chat.send("design a pricing tier"):
            ...

Every omnigent ``StreamHooks`` callback maps to one MLflow span
boundary. The result is a faithful trace of agent turns, tool calls,
sub-agent delegations, reasoning blocks, and context compaction.
All spans nest under one trace per agent turn, visible in the MLflow
Traces UI alongside any other instrumented workload.
"""

from importlib.metadata import PackageNotFoundError, version

from .hooks import OmnigentMlflowHooks

try:
    __version__ = version("omnigent-mlflow")
except PackageNotFoundError:
    # Editable install / running from source tree without a wheel.
    __version__ = "0.0.0+local"

__all__ = ["OmnigentMlflowHooks", "__version__"]
