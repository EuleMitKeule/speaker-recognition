# Changelog

## Unreleased

- Fix the add-on so it builds when the Home Assistant Supervisor builds it from
  the repository. It previously failed with `"/pyproject.toml": not found` /
  `"/speaker_recognition": not found`, because those sources are only copied into
  the add-on build context by CI (`sync-addon-sources.sh`), not by the Supervisor.
- Base the add-on on the published server image (built `FROM python:3.9-slim`)
  instead of an Alpine base. The server pins `resemblyzer` / `torch` to
  `python_version < '3.10'`, so on the Alpine base (Python 3.12) those ML
  dependencies were skipped and the server could not run.
- Replace the s6-overlay / bashio service with a small POSIX `run.sh` entrypoint
  that maps the add-on options to the server's environment variables.
- Add a `Validate add-on` CI workflow that builds the add-on image (and smoke
  tests the import) on every push and pull request.
