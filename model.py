"""
MPI runtime model: fit and compare against Extra-P.

Fits T(P, S) = L*(1 + log2(P)) + G*(log2(P) + sqrt(log2(P)))*S
with non-negative least squares on the training data only, then compares
it with the Extra-P predictions on both the training and the external
test datasets (metrics, per-row errors and node-wise bar charts).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import nnls
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "extrap_default_predictions.csv"
TEST_FILE = "extrap_external_predictions.csv"

OUTPUT_DIR = Path("mpi_model_comparison")

NODE_COL = "nodes"
SIZE_COL = "msg_bytes"
ACTUAL_COL = "actual_us"
EXTRAP_COL = "extrap_pred_us"


# ============================================================
# MPI MODEL
# ============================================================

def A(P):
    return 1.0 + np.log2(P)


def B(P):
    logp = np.log2(P)
    return logp + np.sqrt(logp)


def design_matrix(df):
    P = df[NODE_COL].to_numpy(dtype=float)
    S = df[SIZE_COL].to_numpy(dtype=float)

    return np.column_stack([
        A(P),
        B(P) * S,
    ])


def predict_own_model(df, L, G):
    return design_matrix(df) @ np.array([L, G])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    nonzero = np.abs(actual) > 1e-12

    return {
        "MAE_us": mean_absolute_error(actual, predicted),
        "RMSE_us": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2 else np.nan
        ),
        "MAPE_percent": (
            np.mean(
                np.abs(
                    (actual[nonzero] - predicted[nonzero])
                    / actual[nonzero]
                )
            ) * 100
            if nonzero.any() else np.nan
        ),
    }


def add_prediction_errors(df, prediction_col, prefix):
    df[f"{prefix}_error_us"] = df[prediction_col] - df[ACTUAL_COL]
    df[f"{prefix}_abs_error_us"] = df[f"{prefix}_error_us"].abs()
    df[f"{prefix}_ape_percent"] = (
        df[f"{prefix}_abs_error_us"] / df[ACTUAL_COL].abs() * 100
    )


# ============================================================
# PLOTTING
# ============================================================

def message_label(size):
    if size >= 1024**2:
        return f"{size / 1024**2:g} MiB"
    return f"{size / 1024:g} KiB"


def plot_nodewise(df, dataset_name):
    plot_dir = OUTPUT_DIR / f"{dataset_name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for node, group in df.groupby(NODE_COL):
        group = group.sort_values(SIZE_COL).reset_index(drop=True)

        x = np.arange(len(group))
        width = 0.26

        fig, ax = plt.subplots(figsize=(15, 7))

        ax.bar(x - width, group[ACTUAL_COL], width,
               label="Actual Runtime")
        ax.bar(x, group[EXTRAP_COL], width,
               label="Extra-P Prediction")
        ax.bar(x + width, group["own_pred_us"], width,
               label="Your MPI Model")

        labels = [message_label(s) for s in group[SIZE_COL]]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

        ax.set_xlabel("Message Size")
        ax.set_ylabel("Runtime (µs)")
        ax.set_title(
            f"Actual vs Extra-P vs MPI Model — {int(node)} Nodes"
        )

        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        output_file = plot_dir / f"node_{int(node)}_three_bar.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved three-bar graphs to: {plot_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    required = {NODE_COL, SIZE_COL, ACTUAL_COL, EXTRAP_COL}

    for name, df in [("Training", train), ("External", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} CSV missing columns: {sorted(missing)}"
            )

        df.dropna(
            subset=[NODE_COL, SIZE_COL, ACTUAL_COL, EXTRAP_COL],
            inplace=True,
        )

        df = df[
            (df[NODE_COL] > 0)
            & (df[SIZE_COL] > 0)
            & (df[ACTUAL_COL] > 0)
        ]

        if name == "Training":
            train = df.copy()
        else:
            test = df.copy()

    # Ensure each node/message-size observation is unique.
    for name, df in [("Training", train), ("External", test)]:
        if df.duplicated([NODE_COL, SIZE_COL]).any():
            raise ValueError(
                f"{name} data has duplicate node/message-size pairs."
            )

    print("=" * 70)
    print("TRAINING YOUR MPI MODEL")
    print("=" * 70)
    print(f"Training observations: {len(train)}")
    print(f"External test observations: {len(test)}")

    # Fit ONLY on the internal dataset.
    X_train = design_matrix(train)
    y_train = train[ACTUAL_COL].to_numpy(dtype=float)

    params, residual_norm = nnls(X_train, y_train)
    L, G = params

    print("\nYour fitted model:")
    print("T(P,S) = L*(1 + log2(P))")
    print("       + G*(log2(P) + sqrt(log2(P)))*S")
    print(f"L = {L:.10f} µs")
    print(f"G = {G:.12e} µs/byte")
    print(f"NNLS residual norm = {residual_norm:.6f}")

    # Predictions on both datasets.
    train["own_pred_us"] = predict_own_model(train, L, G)
    test["own_pred_us"] = predict_own_model(test, L, G)

    # Error columns for both models.
    for df in [train, test]:
        add_prediction_errors(df, "own_pred_us", "own")
        add_prediction_errors(df, EXTRAP_COL, "extrap")

    # Metrics: training and external test, independently.
    rows = []

    for split_name, df in [
        ("Training", train),
        ("External test", test),
    ]:
        for model_name, pred_col in [
            ("Your MPI model", "own_pred_us"),
            ("Extra-P", EXTRAP_COL),
        ]:
            metrics = calculate_metrics(df[ACTUAL_COL], df[pred_col])

            rows.append({
                "dataset": split_name,
                "model": model_name,
                "observations": len(df),
                **metrics,
            })

    metrics_df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(metrics_df.to_string(index=False, float_format="%.4f"))

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Save detailed predictions and errors.
    train.to_csv(OUTPUT_DIR / "training_comparison.csv", index=False)
    test.to_csv(OUTPUT_DIR / "external_test_comparison.csv", index=False)

    # Node-wise plots for each dataset.
    plot_nodewise(train, "training")
    plot_nodewise(test, "external_test")

    print("\nCompleted.")
    print(f"Results directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()