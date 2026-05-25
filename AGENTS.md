# playground — Kaggle Playground Series

## Project setup

```sh
uv sync                  # sync deps
uv add <package>         # add a dependency
uv run <command>         # run in the venv
```

The kaggle CLI is invoked as `uv run kaggle ...`.

## Tooling

use polars for data processing. altair for plotting. 

## Submitting to Kaggle

```sh
uv run kaggle competitions submit {competition} -f {folder}/submission.csv -m "message"
```

Check latest submission score:
```sh
uv run kaggle competitions submissions {competition} --csv | head -5
```

## Quick start (just)

```sh
just                # list available recipes
just comp s6e5      # full bootstrap: folder + download + template + open
just nb foo.py      # open s6e5/foo.py (creates from template if missing)
```

