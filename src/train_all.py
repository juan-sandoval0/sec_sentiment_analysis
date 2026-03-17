"""
train_all.py - builds features, runs grid search for all models, saves val results
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.features import build_all_features
from src.models import (ridge_grid_search, logreg_grid_search,
                        rf_grid_search, mlp_grid_search)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

feature_sets, targets, df_train, df_val, df_test = build_all_features()

y_reg_train, y_reg_val, y_reg_test = targets["regression"]
y_cls_train, y_cls_val, y_cls_test = targets["classification"]

print("\nRidge")
ridge_results, ridge_models = [], {}
for fs_name, (X_tr, X_val, X_te) in feature_sets.items():
    df_res, best = ridge_grid_search(X_tr, y_reg_train, X_val, y_reg_val,
                                     feature_set_name=fs_name)
    ridge_results.append(df_res)
    ridge_models[fs_name] = best

ridge_df = pd.concat(ridge_results, ignore_index=True)
print(ridge_df.sort_values(["feature_set", "val_rmse"]).to_string(index=False))

print("\nLogistic Regression")
logreg_results, logreg_models = [], {}
for fs_name, (X_tr, X_val, X_te) in feature_sets.items():
    df_res, best = logreg_grid_search(X_tr, y_cls_train, X_val, y_cls_val,
                                      feature_set_name=fs_name)
    logreg_results.append(df_res)
    logreg_models[fs_name] = best

logreg_df = pd.concat(logreg_results, ignore_index=True)
print(logreg_df.sort_values(["feature_set", "val_auc"], ascending=[True, False])
              .to_string(index=False))

print("\nRandom Forest")
rf_results, rf_models = [], {}
for fs_name, (X_tr, X_val, X_te) in feature_sets.items():
    df_res, best = rf_grid_search(X_tr, y_cls_train, X_val, y_cls_val,
                                  task="classification", feature_set_name=fs_name)
    rf_results.append(df_res)
    rf_models[fs_name] = best

rf_df = pd.concat(rf_results, ignore_index=True)
print(rf_df.sort_values(["feature_set", "val_auc"], ascending=[True, False])
           .to_string(index=False))

print("\nMLP")
mlp_results, mlp_models, mlp_histories = [], {}, {}
for fs_name in ["financial", "sentiment", "tfidf", "finbert"]:
    X_tr, X_val, X_te = feature_sets[fs_name]
    df_res, best, hist = mlp_grid_search(X_tr, y_cls_train, X_val, y_cls_val,
                                         task="classification",
                                         feature_set_name=fs_name)
    mlp_results.append(df_res)
    mlp_models[fs_name]    = best
    mlp_histories[fs_name] = hist

mlp_df = pd.concat(mlp_results, ignore_index=True)
print(mlp_df.sort_values(["feature_set", "val_auc"], ascending=[True, False])
            .to_string(index=False))

summary_rows = []
for fs in feature_sets:
    for df_, label in [(logreg_df, "logreg"), (rf_df, "rf"), (mlp_df, "mlp")]:
        sub = df_[df_["feature_set"] == fs]
        if len(sub):
            best = sub.loc[sub["val_auc"].idxmax()]
            summary_rows.append({"model": label, "feature_set": fs,
                                  "val_auc": round(best["val_auc"], 4),
                                  "val_f1":  round(best["val_f1"],  4)})

summary_df = pd.DataFrame(summary_rows)
print("\n", summary_df.sort_values("val_auc", ascending=False).to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS_DIR, "val_summary.csv"), index=False)
print("saved results/val_summary.csv")
