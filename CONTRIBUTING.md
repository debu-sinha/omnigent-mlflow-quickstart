# Contributing

Thanks for the interest. PRs and issues are welcome.

## Quick start

```bash
git clone https://github.com/debu-sinha/omnigent-mlflow-quickstart.git
cd omnigent-mlflow-quickstart
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,omnigent,genai]"
pre-commit install
```

## Running checks

```bash
ruff check .
ruff format --check .
pytest -q
```

CI runs the same checks on Python 3.12 and 3.13. If they're clean
locally they'll be clean on CI.

## Filing an issue

Use one of the templates under `.github/ISSUE_TEMPLATE/`. Bug reports
should include the omnigent version, MLflow version, Python version,
and a minimum repro. If you don't have a clean repro yet, file the
issue anyway with what you've got.

## Sending a PR

- Branch from `main`. Keep the diff focused. One conceptual change
  per PR.
- Add tests when adding behavior. The fixture under `tests/conftest.py`
  uses an in-memory SQLite tracking store so you don't need network
  access for the suite.
- Sign your commit (`git commit -s`) for the DCO.
- Run `pre-commit run --all-files` before pushing so the bot doesn't
  catch you on style.

## Pre-merge checklist

- [ ] Tests pass locally and on CI.
- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] If behavior changed, the README or docstrings reflect it.
- [ ] If a public API moved, the CHANGELOG has an `## [Unreleased]`
      entry under the right heading.

## Project layout

```
src/omnigent_mlflow/
  hooks.py     # OmnigentMlflowHooks, the StreamHooks -> MLflow span adapter
  judges.py    # Two illustrative judge.align-compatible scorers
examples/
  trace_debby.py    # End-to-end against the bundled debby agent
  debate_judge.py   # Score the resulting trace
tests/
  test_hooks.py     # Unit tests for span lifecycle pairing
docs/
  diagrams/architecture.svg
  logo.svg
```

## Releasing

Pushing a tag matching `v*` triggers `.github/workflows/publish.yml`
which builds and uploads to PyPI via OIDC trusted publishing.
Maintainers do this after merging the relevant CHANGELOG version
header.
