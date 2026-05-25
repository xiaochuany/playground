# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "marimo",
#   "polars",
# ]
# ///

import marimo

__generated_with = "marimo"
app = marimo.App()


@app.cell
def __(mo):
    import polars as pl

    train = pl.scan_csv("train.csv")
    test = pl.scan_csv("test.csv")
    sub = pl.scan_csv("sample_submission.csv")
    return pl, sub, test, train


@app.cell
def __(train):
    train.head(3).collect()
    return


@app.cell
def __(train):
    train.describe().collect()
    return


if __name__ == "__main__":
    app.run()
