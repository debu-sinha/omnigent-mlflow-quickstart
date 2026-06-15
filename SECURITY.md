# Security Policy

## Supported versions

Alpha. The `main` branch is the supported line; older snapshots aren't
maintained.

## Reporting a vulnerability

Please don't open a public issue. Send a private security advisory
through GitHub:
<https://github.com/debu-sinha/omnigent-mlflow-quickstart/security/advisories/new>

Reports get acknowledged within 72 hours. Fixes for confirmed issues
land on `main` and a tagged release within two weeks unless the issue
is non-trivial, in which case I'll update the reporter on the
timeline.

## Out of scope

This adapter is a thin layer over MLflow's tracing primitives and
omnigent's `StreamHooks` surface. Vulnerabilities in those underlying
libraries should go to their projects directly:

- MLflow: <https://github.com/mlflow/mlflow/security/policy>
- omnigent: <https://github.com/omnigent-ai/omnigent/security/policy>

Don't include live API keys, customer data, or workspace URLs in any
report.
