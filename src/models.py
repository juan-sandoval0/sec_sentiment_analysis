"""
models.py - ridge, logistic regression, random forest, MLP with manual grid search
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(np.mean(np.abs(y_true - y_pred))),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1":  float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


# ridge regression

# spanning five decades: low end for financial/sentiment (p<<n),
# high end for tfidf/finbert where X'X is rank-deficient
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]


def ridge_grid_search(X_train, y_train, X_val, y_val,
                      feature_set_name: str = "") -> tuple[pd.DataFrame, Ridge]:
    records    = []
    best_rmse  = np.inf
    best_model = None

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        tr_m  = regression_metrics(y_train, model.predict(X_train))
        val_m = regression_metrics(y_val,   model.predict(X_val))

        records.append({
            "feature_set": feature_set_name,
            "alpha":       alpha,
            "train_rmse":  tr_m["rmse"], "val_rmse": val_m["rmse"],
            "train_r2":    tr_m["r2"],   "val_r2":   val_m["r2"],
            "train_mae":   tr_m["mae"],  "val_mae":  val_m["mae"],
        })

        if val_m["rmse"] < best_rmse:
            best_rmse  = val_m["rmse"]
            best_model = model

    return pd.DataFrame(records), best_model


# logistic regression

# C = 1/lambda. strong regularization needed for tfidf/finbert (p>>n)
LOGREG_CS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def logreg_grid_search(X_train, y_train, X_val, y_val,
                       feature_set_name: str = "") -> tuple[pd.DataFrame, LogisticRegression]:
    """class_weight='balanced' because we have a 25/75 class split"""
    records   = []
    best_auc  = -np.inf
    best_model = None

    for C in LOGREG_CS:
        model = LogisticRegression(C=C, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")
        model.fit(X_train, y_train)

        tr_prob  = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]

        tr_m  = classification_metrics(y_train, tr_prob)
        val_m = classification_metrics(y_val,   val_prob)

        records.append({
            "feature_set": feature_set_name,
            "C":           C,
            "train_auc":   tr_m["auc"], "val_auc": val_m["auc"],
            "train_f1":    tr_m["f1"],  "val_f1":  val_m["f1"],
            "train_acc":   tr_m["acc"], "val_acc": val_m["acc"],
        })

        if val_m["auc"] > best_auc:
            best_auc   = val_m["auc"]
            best_model = model

    return pd.DataFrame(records), best_model


# random forest

# 2 x 3 x 3 = 18 combos
# depth 3 = max 8 leaves (conservative for n~200), depth 10 probably overfits
# min_samples_leaf=15 ~ sqrt(216), common rule of thumb
RF_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5, 10],
    "min_samples_leaf":[1, 10, 15],
}


def rf_grid_search(X_train, y_train, X_val, y_val,
                   task: str = "classification",
                   feature_set_name: str = "") -> tuple[pd.DataFrame, object]:
    combos = [
        (n, d, m)
        for n in RF_GRID["n_estimators"]
        for d in RF_GRID["max_depth"]
        for m in RF_GRID["min_samples_leaf"]
    ]

    records    = []
    best_score = -np.inf   # negate RMSE so higher is always better
    best_model = None

    for n_est, max_depth, min_leaf in tqdm(
            combos, desc=f"RF ({feature_set_name}, {task})", leave=False):

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth,
                min_samples_leaf=min_leaf, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)

            tr_m  = classification_metrics(y_train, model.predict_proba(X_train)[:, 1])
            val_m = classification_metrics(y_val,   model.predict_proba(X_val)[:, 1])
            score = val_m["auc"]

            records.append({
                "feature_set": feature_set_name,
                "n_estimators": n_est, "max_depth": str(max_depth),
                "min_samples_leaf": min_leaf,
                "train_auc": tr_m["auc"], "val_auc": val_m["auc"],
                "train_f1":  tr_m["f1"],  "val_f1":  val_m["f1"],
            })
        else:
            model = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth,
                min_samples_leaf=min_leaf, random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)

            tr_m  = regression_metrics(y_train, model.predict(X_train))
            val_m = regression_metrics(y_val,   model.predict(X_val))
            score = -val_m["rmse"]   # negate so higher = better

            records.append({
                "feature_set": feature_set_name,
                "n_estimators": n_est, "max_depth": str(max_depth),
                "min_samples_leaf": min_leaf,
                "train_rmse": tr_m["rmse"], "val_rmse": val_m["rmse"],
                "train_r2":   tr_m["r2"],   "val_r2":   val_m["r2"],
            })

        if score > best_score:
            best_score = score
            best_model = model

    return pd.DataFrame(records), best_model


# MLP

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers   = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# 3 x 2 x 2 x 3 = 36 combos
MLP_GRID = {
    "hidden_dims":  [[64, 32], [128, 64], [256, 128, 64]],
    "dropout":      [0.2, 0.5],
    "lr":           [3e-4, 1e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2],
}


def train_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims: list, dropout: float, lr: float, weight_decay: float,
    task: str = "classification",
    max_epochs: int = 50, batch_size: int = 64,
    patience: int = 7, device=None,
) -> tuple[MLP, dict]:
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    model     = MLP(X_train.shape[1], hidden_dims, dropout).to(device)
    criterion = (nn.BCEWithLogitsLoss() if task == "classification"
                 else nn.MSELoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    stopper   = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(max_epochs):

        model.train()
        batch_losses = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # avoid explosion
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss   = criterion(val_logits, y_v).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    history["epochs_trained"] = epoch + 1

    return model, history


def mlp_grid_search(
    X_train, y_train, X_val, y_val,
    task: str = "classification",
    feature_set_name: str = "",
) -> tuple[pd.DataFrame, MLP, dict]:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    combos = [
        (hd, dr, lr, wd)
        for hd in MLP_GRID["hidden_dims"]
        for dr in MLP_GRID["dropout"]
        for lr in MLP_GRID["lr"]
        for wd in MLP_GRID["weight_decay"]
    ]

    records      = []
    best_score   = -np.inf
    best_model   = None
    best_history = None

    for hidden_dims, dropout, lr, wd in tqdm(
            combos, desc=f"MLP grid ({feature_set_name}, {task})"):

        model, history = train_mlp(
            X_train, y_train, X_val, y_val,
            hidden_dims=hidden_dims, dropout=dropout,
            lr=lr, weight_decay=wd,
            task=task, device=device,
        )

        model.eval()
        with torch.no_grad():
            tr_logits  = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
            val_logits = model(torch.tensor(X_val,   dtype=torch.float32).to(device)).cpu().numpy()

        if task == "classification":
            tr_prob  = 1 / (1 + np.exp(-tr_logits))   # manual sigmoid
            val_prob = 1 / (1 + np.exp(-val_logits))
            tr_m     = classification_metrics(y_train, tr_prob)
            val_m    = classification_metrics(y_val,   val_prob)
            score    = val_m["auc"]

            records.append({
                "feature_set":   feature_set_name,
                "hidden_dims":   str(hidden_dims),
                "dropout":       dropout,
                "lr":            lr,
                "weight_decay":  wd,
                "epochs_trained": history["epochs_trained"],
                "train_auc": tr_m["auc"], "val_auc": val_m["auc"],
                "train_f1":  tr_m["f1"],  "val_f1":  val_m["f1"],
            })
            is_better = score > best_score
        else:
            tr_m  = regression_metrics(y_train, tr_logits)
            val_m = regression_metrics(y_val,   val_logits)
            score = -val_m["rmse"]

            records.append({
                "feature_set":   feature_set_name,
                "hidden_dims":   str(hidden_dims),
                "dropout":       dropout,
                "lr":            lr,
                "weight_decay":  wd,
                "epochs_trained": history["epochs_trained"],
                "train_rmse": tr_m["rmse"], "val_rmse": val_m["rmse"],
                "train_r2":   tr_m["r2"],   "val_r2":   val_m["r2"],
            })
            is_better = score > best_score

        if is_better:
            best_score   = score
            best_model   = model
            best_history = history

    return pd.DataFrame(records), best_model, best_history
