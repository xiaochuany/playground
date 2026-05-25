# playground — Kaggle Playground Series

This file is loaded by pi at the start of every session in this project.

## Project setup

Managed with `uv`. Python 3.14, virtual env in `.venv/`.

```sh
uv sync                  # sync deps
uv add <package>         # add a dependency
uv run <command>         # run in the venv
```

Key dependencies: `polars`, `marimo`, `kaggle` (all in `pyproject.toml`).
The kaggle CLI is invoked as `uv run kaggle ...`.

## Structure

Each competition gets a directory: `s{season}e{episode}/`.

```
playground/
├── _template/           # starter notebook template
│   └── nb.py
├── s6e5/                # current episode
│   ├── train.csv        (gitignored)
│   ├── test.csv         (gitignored)
│   ├── sample_submission.csv  (gitignored)
│   └── nb1.py           # created from template
├── pyproject.toml
├── justfile
└── AGENTS.md
```

## Quick start (just)

```sh
just                # list available recipes
just comp s6e5      # full bootstrap: folder + download + template + open
just nb foo.py      # open s6e5/foo.py (creates from template if missing)
```

### `just comp <name>` — bootstrap a new competition

Creates the folder, downloads train/test/sample_submission from Kaggle,
copies the starter template, and opens it in marimo — all in one command.

### `just nb` — open a notebook

Copies `_template/nb.py` into the target folder if the file doesn't exist yet,
then opens it in marimo with `--watch --no-token`.

The `folder` variable at the top of `justfile` controls which episode directory
is used by `just nb`. Change it when switching between episodes.

## Workflow

1. `just comp s6e6` — bootstraps everything
2. `just nb nb2.py` — create and open additional experiments
3. Submit via kaggle CLI: `uv run kaggle competitions submit ...`
