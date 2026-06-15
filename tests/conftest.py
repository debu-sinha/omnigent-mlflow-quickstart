"""Shared test fixtures.

The local_mlflow autouse fixture points MLflow at an in-memory
SQLite database per test so the suite needs no network access and
leaves no files behind. MLflow 3.x removed the file-store tracking
backend from the supported set, so SQLite is the recommended local
option.
"""

from __future__ import annotations

import mlflow
import pytest


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path, monkeypatch):
    db = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tmp_path
