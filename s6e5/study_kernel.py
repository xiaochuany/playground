import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return mo,


@app.cell
def _(mo):
    mo.md(
        """
        # Key Ideas from PS S6 E5 Top Kernel (RealMLP + PyTabKit)

        Public LB: **~0.9549** (vs our HGB 0.8746)

        ## 1. External data is the biggest lever

        The kernel loads the original F1 strategy dataset from Kaggle
        (`f1_strategy_dataset_v4.csv`) and **concatenates it with the training
        data inside each CV fold**. This more than doubles the training set.
        The Playground data is synthetic — the original data is the real signal.

        ## 2. Tabular deep learning beats tree models here

        Uses `RealMLP_TD_Classifier` from pytabkit — a modern MLP designed for
        tabular data. Architecture: [512, 256, 128] with SiLU activations,
        embedding size 6 for categoricals, PLR (Periodic Linear Representation)
        for numericals. Only 5 epochs, but 20 ensembles per fold → 100 models
        total. This is *not* a tree — it's a carefully regularized neural net.

        ## 3. Feature engineering still matters (even for deep learning)

        Despite using a neural net, the kernel engineers features:
        - **Arithmetic interactions**: LapNumber / RaceProgress, TyreLife /
          LapNumber, LapTime × Cumulative_Degradation
        - **Binning**: discretizes RaceProgress and LapTime into categorical
          buckets
        - **Cross interactions**: (Race, Compound) and (Race, Year) as new
          categorical features
        - **Count encoding**: frequency of each category
        - **Target encoding** on the interaction features (Race×Compound,
          Race×Year)

        ## 4. Proper categorical handling

        - Low-cardinality cats get one-hot encoded (max 18 unique values)
        - Higher-cardinality cats get learned embeddings (size 6)
        - All categoricals are properly declared to the model

        ## 5. Regularization is the secret sauce

        - Dropout with scheduling (invsqrtp1e-3)
        - Label smoothing with cosine schedule
        - Weight decay 0.01
        - Learning rate scheduling (lin_cos_log_15)
        - Bias initialization: negative uniform dynamic

        ## 6. Takeaways for our benchmark

        | What we did | What the top kernel does |
        |---|---|
        | No external data | Adds original F1 dataset |
        | HistGBM (trees) | RealMLP (deep learning) |
        | Minimal features | ~30 engineered features |
        | Single model | 100-model ensemble |
        | 0.8746 public | 0.9549 public |
        """
    )
    return


if __name__ == "__main__":
    app.run()
