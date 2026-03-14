"""
run_test_eval.py

trains the best model × feature-set combos on train+val,
then finally evaluates on the held-out 2020-2023 test set.

models:
    logreg × tfidf, logreg × finbert
    rf     × tfidf, rf     × finbert
    mlp    × tfidf, mlp    × finbert
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.features import build_all_features
from src.models import logreg_grid_search, rf_grid_search, mlp_grid_search
from src.evaluate import plot_roc_curves

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_TO_EVAL = [
    ("logreg", "tfidf"),
    ("logreg", "finbert"),
    ("rf",     "tfidf"),
    ("rf",     "finbert"),
    ("mlp",    "tfidf"),
    ("mlp",    "finbert"),
]

# ── Features ───────────────────────────────────────────────────────────────────
print("Loading features...")
feature_sets, targets, df_train, df_val, df_test = build_all_features()
y_cls_train, y_cls_val, y_cls_test = targets["classification"]
print(f"  Train={len(y_cls_train)}  Val={len(y_cls_val)}  Test={len(y_cls_test)}\n")

assert len(y_cls_test) > 0, "Test set is empty — check data pipeline."

# ── Train & evaluate ───────────────────────────────────────────────────────────
test_results = []

for model_type, fs_name in MODELS_TO_EVAL:
    X_tr, X_val, X_te = feature_sets[fs_name]
    label = f"{model_type} ({fs_name})"
    print(f"Training {label}...")

    if model_type == "logreg":
        df_res, best_model = logreg_grid_search(
            X_tr, y_cls_train, X_val, y_cls_val, feature_set_name=fs_name)
        y_prob_val  = best_model.predict_proba(X_val)[:, 1]
        y_prob_test = best_model.predict_proba(X_te)[:, 1]

    elif model_type == "rf":
        df_res, best_model = rf_grid_search(
            X_tr, y_cls_train, X_val, y_cls_val,
            task="classification", feature_set_name=fs_name)
        y_prob_val  = best_model.predict_proba(X_val)[:, 1]
        y_prob_test = best_model.predict_proba(X_te)[:, 1]

    elif model_type == "mlp":
        df_res, best_model, _ = mlp_grid_search(
            X_tr, y_cls_train, X_val, y_cls_val,
            task="classification", feature_set_name=fs_name)
        best_model = best_model.cpu().eval()
        with torch.no_grad():
            y_prob_val  = torch.sigmoid(best_model(
                torch.tensor(X_val, dtype=torch.float32))).numpy()
            y_prob_test = torch.sigmoid(best_model(
                torch.tensor(X_te,  dtype=torch.float32))).numpy()

    auc_val  = roc_auc_score(y_cls_val,  y_prob_val)
    auc_test = roc_auc_score(y_cls_test, y_prob_test)
    f1_val   = f1_score(y_cls_val,  (y_prob_val  >= 0.5).astype(int), zero_division=0)
    f1_test  = f1_score(y_cls_test, (y_prob_test >= 0.5).astype(int), zero_division=0)
    acc_test = accuracy_score(y_cls_test, (y_prob_test >= 0.5).astype(int))

    test_results.append({
        "model":       model_type,
        "feature_set": fs_name,
        "val_auc":     round(auc_val,  4),
        "test_auc":    round(auc_test, 4),
        "auc_delta":   round(auc_test - auc_val, 4),  # positive = generalization bonus
        "val_f1":      round(f1_val,   4),
        "test_f1":     round(f1_test,  4),
        "test_acc":    round(acc_test, 4),
    })
    print(f"  val AUC={auc_val:.4f}  test AUC={auc_test:.4f}  "
          f"test F1={f1_test:.4f}  test acc={acc_test:.4f}")

# ── Results table ──────────────────────────────────────────────────────────────
test_df = pd.DataFrame(test_results).sort_values("test_auc", ascending=False)

print("\n" + "=" * 75)
print("TEST SET RESULTS  (2020–2023)")
print("=" * 75)
print(test_df.to_string(index=False))
print("=" * 75)
print(f"\nMean test AUC          : {test_df['test_auc'].mean():.4f}")
print(f"Best test AUC          : {test_df['test_auc'].max():.4f}  "
      f"({test_df.loc[test_df['test_auc'].idxmax(), 'model']} × "
      f"{test_df.loc[test_df['test_auc'].idxmax(), 'feature_set']})")
print(f"Mean AUC drop val→test : {test_df['auc_delta'].mean():.4f}")

test_df.to_csv(os.path.join(RESULTS_DIR, "test_summary.csv"), index=False)
print("\nSaved → results/test_summary.csv")
