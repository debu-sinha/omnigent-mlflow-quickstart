<!-- Keep this concise. Reviewers skim. If it's longer than 200 words you're probably hiding something. -->

## Summary

<!-- What changed and why, in 1-3 bullets or a short paragraph. -->

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor / chore
- [ ] Docs
- [ ] Test / CI
- [ ] Breaking change

## Test coverage

- [ ] Unit tests added or updated
- [ ] Integration / end-to-end checked locally against an omnigent server
- [ ] Existing tests still cover this change
- [ ] Not applicable

## Coverage rationale

<!-- Describe the exact commands run and the coverage added or updated.
     If you didn't add tests, explain why the existing coverage suffices
     or why tests aren't applicable. Reviewers will ask if this is blank. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `pytest -q` passes locally
- [ ] If a public API moved, the CHANGELOG has an `## [Unreleased]` entry
- [ ] If README behavior changed, README is updated
- [ ] Commit is signed off (`git commit -s`) for the DCO
