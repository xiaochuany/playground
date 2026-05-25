# ── Playground Series — quick recipes ──────────────────────────────────

folder := "s6e5"

# list available recipes (this is the default when you run `just`)
default:
    @just --list

# bootstrap a new competition: create folder, download data, open notebook
comp name:
    mkdir -p {{name}}
    uv run kaggle competitions download playground-series-{{name}} \
      -f train.csv -f test.csv -f sample_submission.csv -p {{name}}
    cp _template/nb.py {{name}}/nb1.py
    uv run marimo edit {{name}}/nb1.py --watch --no-token

# open (or create from template, then open) a marimo notebook
nb path:
    test -f {{folder}}/{{path}} || cp _template/nb.py {{folder}}/{{path}}
    uv run marimo edit {{folder}}/{{path}} --watch --no-token
