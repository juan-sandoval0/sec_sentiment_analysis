"""
evaluate.py

plotting and results table stuff.
all figures saved to results/figures/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))


# ── Target distribution ────────────────────────────────────────────────────────

def plot_target_distribution(y_train: np.ndarray, y_val: np.ndarray,
                              threshold: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_train, bins=40, alpha=0.7, label="Train",
            color="steelblue", density=True)
    ax.hist(y_val,   bins=40, alpha=0.7, label="Val",
            color="coral",     density=True)
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"75th pct threshold = {threshold:.3f}")
    ax.set_xlabel("Realized Volatility (annualized)")
    ax.set_ylabel("Density")
    ax.set_title("Target Distribution: Realized Volatility")
    ax.legend()
    fig.tight_layout()
    _save(fig, "target_distribution.png")


# ── Regularization sensitivity curves ─────────────────────────────────────────

def plot_regularization_curves(results_df: pd.DataFrame,
                                param_col: str,
                                metric_col: str,
                                title: str,
                                fname: str):
    """
    two panels: train metric (left) and val metric (right) vs. hyperparam,
    one curve per feature set. works for alpha, C, or any categorical param.
    """
    train_col    = metric_col.replace("val_", "train_")
    feature_sets = results_df["feature_set"].unique()
    colors       = plt.cm.tab10(np.linspace(0, 1, len(feature_sets)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for col, ax, ylabel in zip(
        [train_col, metric_col],
        axes,
        [f"Train {metric_col.split('_',1)[1].upper()}",
         f"Val   {metric_col.split('_',1)[1].upper()}"],
    ):
        for fs, color in zip(feature_sets, colors):
            sub = results_df[results_df["feature_set"] == fs].sort_values(param_col)
            x   = sub[param_col].astype(str)
            ax.plot(x, sub[col], marker="o", label=fs, color=color)
        ax.set_xlabel(param_col)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, fname)


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(model_probs: dict, y_true: np.ndarray,
                    fname: str = "roc_curves.png"):
    """
    model_probs: {model_label: predicted_probability_array}
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_probs)))
    fig, ax = plt.subplots(figsize=(7, 6))

    for (name, y_prob), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc(fpr,tpr):.3f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Validation Set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, fname)


# ── MLP training curves ────────────────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = "MLP Training Curves",
                          fname: str = "mlp_training.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss",   color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["lr"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule (ReduceLROnPlateau)")
    axes[1].set_yscale("log")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save(fig, fname)


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list,
                             top_n: int = 20, fname: str = "rf_importance.png"):
    importances = model.feature_importances_
    idx  = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Random Forest Feature Importance (top {top_n})")
    fig.tight_layout()
    _save(fig, fname)


def plot_ridge_coefficients(model, feature_names: list,
                             top_n: int = 20, fname: str = "ridge_coefs.png"):
    coefs  = model.coef_
    idx    = np.argsort(np.abs(coefs))[-top_n:]
    colors = ["steelblue" if c > 0 else "coral" for c in coefs[idx]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], coefs[idx], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Ridge Regression Coefficients (top {top_n} by |magnitude|)")
    fig.tight_layout()
    _save(fig, fname)


# ── Heatmap: val AUC across feature sets × model families ─────────────────────

def plot_model_comparison_heatmap(summary_df: pd.DataFrame,
                                   metric: str = "val_auc",
                                   fname: str = "model_comparison.png"):
    """
    summary_df needs columns: model, feature_set, <metric>
    """
    pivot = summary_df.pivot(index="feature_set", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Validation {metric.upper()} by Model × Feature Set")
    fig.tight_layout()
    _save(fig, fname)


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(all_results: dict) -> pd.DataFrame:
    """
    all_results: {model_label: {metric_name: value, ...}}
    prints a formatted summary table and returns it as a DataFrame
    """
    rows = [{"model": name, **metrics} for name, metrics in all_results.items()]
    df   = pd.DataFrame(rows)
    print("\n" + "=" * 65)
    print("VALIDATION SET RESULTS SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)
    return df


# ── Internal ───────────────────────────────────────────────────────────────────

def _save(fig, fname: str):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
