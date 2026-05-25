import marimo

__generated_with = "marimo"
app = marimo.App()


@app.cell
def __(mo):
    from pathlib import Path
    import polars as pl

    data_dir = Path(__file__).resolve().parent
    train = pl.read_csv(data_dir / "train.csv")
    test = pl.read_csv(data_dir / "test.csv")
    sub = pl.read_csv(data_dir / "sample_submission.csv")
    return pl, sub, test, train


@app.cell
def __(train):
    train.head(3)
    return


@app.cell
def __(train):
    train.describe()
    return


if __name__ == "__main__":
    app.run()
