# Inkling-2-Small TP4 recipe

This folder is the recipe: `setup.sh` and `setup/`, `release-defaults.sh`, `config.example.sh` and
`hosts.conf.example`, `start.sh`, `stop.sh`, `status.sh`, `rank.sh`, `check.sh`, `bench.sh` (run it from a clone of
the repository: it uses `tools/`), `weights.sh` and `weights/`, `smoke.py` and `prompt_ids.py`, `container/` and
`fabric/`. Changes by version: [CHANGELOG.md](CHANGELOG.md).

How to install and run it, the requirements, the benchmarks, what is checked, the settings and the API are in the
[repository README](../../README.md). Each topic in full: [docs/](docs/) (INSTALL-MANUAL, CORRECTNESS, BENCHMARKS,
SETTINGS).

Quick start, from this folder: `cp hosts.conf.example my-sparks.conf`, set `HOSTS` and `ROOT`, then run the
`./setup.sh` command from the README.
