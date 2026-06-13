"""Drop-in MLflow tracing for omnigent agents.

Usage::

    from omnigent_client import OmnigentClient, BlockStream
    from omnigent_mlflow import OmnigentMlflowHooks

    hooks = OmnigentMlflowHooks(experiment="omnigent-debby")
    async with OmnigentClient(base_url="http://localhost:8080") as c:
        session = c.session(model="debby")
        stream = BlockStream(hooks=hooks.stream_hooks())
        async for block in stream.stream(session, "design a pricing tier"):
            ...

Every omnigent ``StreamHooks`` callback maps to one MLflow span
boundary. The result is a faithful trace of agent turns, tool calls,
sub-agent delegations, reasoning blocks, and context compaction --
visible in the MLflow Traces UI alongside any other instrumented
workload.
"""

from .hooks import OmnigentMlflowHooks

__all__ = ["OmnigentMlflowHooks"]
__version__ = "0.1.0"
