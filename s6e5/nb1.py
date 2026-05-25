import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import polars as pl
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    data_dir = Path(__file__).resolve().parent
    train = pl.read_csv(data_dir / "train.csv")
    test = pl.read_csv(data_dir / "test.csv")

    # --- target encoding with principled smoothing (Beta-Binomial method of moments) ---
    p0 = train["PitNextLap"].mean()  # global rate ≈ 0.199

    def estimate_m(df, col, p0):
        """Estimate prior sample size m via Beta-Binomial method of moments."""
        stats = df.group_by(col).agg(
            pl.col("PitNextLap").mean().alias("mean"),
            pl.len().alias("n"),
        )
        N = stats["n"].sum()
        k = len(stats)
        # Weighted variance of per-category means
        weighted_var = (stats["n"] * (stats["mean"] - p0) ** 2).sum() / N
        # Expected sampling noise (null: all categories share rate p0)
        expected_noise = p0 * (1 - p0) * k / N
        signal = max(0.0, weighted_var - expected_noise)
        # Var(p_i) = p0*(1-p0) / (m+1)  →  m = p0*(1-p0)/Var(p_i) - 1
        m = p0 * (1 - p0) / signal - 1 if signal > 0 else 500.0
        return float(max(1.0, min(1000.0, m)))

    ms = {}
    for col in ["Driver", "Race"]:
        m = estimate_m(train, col, p0)
        ms[col] = m
        print(f"  {col}: estimated m = {m:.1f}")

        stats = train.group_by(col).agg(
            pl.col("PitNextLap").mean().alias(f"{col}_mean"),
            pl.len().alias(f"{col}_count"),
        ).with_columns(
            (
                (pl.col(f"{col}_count") * pl.col(f"{col}_mean") + m * p0)
                / (pl.col(f"{col}_count") + m)
            ).alias(f"{col}_te")
        )
        train = train.join(stats.select(col, f"{col}_te"), on=col)
        test = test.join(stats.select(col, f"{col}_te"), on=col)
    return (
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        StandardScaler,
        make_pipeline,
        pl,
        roc_auc_score,
        test,
        train,
        train_test_split,
    )


@app.cell
def _(
    ColumnTransformer,
    LogisticRegression,
    OneHotEncoder,
    StandardScaler,
    make_pipeline,
    pl,
    roc_auc_score,
    test,
    train,
    train_test_split,
):
    num_cols = [
        "LapNumber",
        "TyreLife",
        "Position",
        "RaceProgress",
        "Position_Change",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "PitStop",
        "Stint",
    ]
    cat_cols = ["Compound"]
    te_cols = ["Driver_te", "Race_te"]

    all_cols = num_cols + cat_cols + te_cols
    X = train.select(all_cols).to_numpy()
    y = train["PitNextLap"].to_numpy()
    X_t = test.select(all_cols).to_numpy()

    n_num = len(num_cols)
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), list(range(n_num))),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                [n_num],
            ),
            (
                "te",
                StandardScaler(),
                list(range(n_num + 1, n_num + 1 + len(te_cols))),
            ),
        ]
    )

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    pipe = make_pipeline(preprocessor, lr)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
    print(f"Validation AUC: {auc:.4f}")

    preds = pipe.predict(X_t)
    test.select("id").with_columns(
        pl.Series("PitNextLap", preds.tolist())
    ).write_csv("s6e5/submission_lr_benchmark.csv")
    print(f"Submission saved. Pos rate: {preds.mean():.3f}")
    return


@app.cell
def _(pl, roc_auc_score, test, train, train_test_split):
    """HistGradientBoosting — handles categoricals natively, no encoding needed"""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Driver_te already computed in cell 1 (cardinality too high for HistGBM native cat support)
    _num = [
        "LapNumber", "TyreLife", "Position", "RaceProgress",
        "Position_Change", "LapTime_Delta", "Cumulative_Degradation",
        "PitStop", "Stint", "Driver_te",
    ]
    _cat = ["Race", "Compound"]

    _X = train.select(_num + _cat).to_numpy()
    _y = train["PitNextLap"].to_numpy()
    _Xt = test.select(_num + _cat).to_numpy()

    _n = len(_num)
    _hgb = HistGradientBoostingClassifier(
        categorical_features=[False] * _n + [True] * len(_cat),
        max_iter=500,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
    )

    _tr, _va, _ytr, _yva = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    _hgb.fit(_tr, _ytr)
    _auc = roc_auc_score(_yva, _hgb.predict_proba(_va)[:, 1])
    print(f"HGB Validation AUC: {_auc:.4f}")

    _preds = _hgb.predict(_Xt)
    test.select("id").with_columns(
        pl.Series("PitNextLap", _preds.tolist())
    ).write_csv("s6e5/submission_hgb_benchmark.csv")
    print(f"HGB submission saved. Pos rate: {_preds.mean():.3f}")
    return


@app.cell
def _():
    print("""
    ## Learnings

    1. Data is synthetic snapshots, not a time series — each row is an independent
       observation. Don't assume sequential consistency within (Driver, Race, Year)
       groups.

    2. Hardcoded feature engineering is fragile — ordinal compound rankings,
       typical stint lengths, and manual interactions didn't beat raw features +
       proper encoding. Let the model learn.

    3. Tree models > linear models for this data — HistGradientBoosting (0.8746
       public AUC) crushed LR (0.7638) because it handles non-linearities and
       interactions without manual engineering.

       Benchmark progression (public AUC):
       - LR + hardcoded features:     0.7551
       - LR + target encoding + OHE:  0.7638
       - HistGradientBoosting:        0.8746

    4. Target encoding with principled smoothing — use Beta-Binomial method of
       moments to estimate m from the data instead of guessing.

    5. Driver matters — adding Driver target encoding lifted LR from 0.755 to
       0.764. Individual driver pit behavior is highly informative.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
