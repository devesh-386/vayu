"""Models compared in this project.

  Persistence   - tomorrow's AQI = today's AQI. The baseline every model must beat.
  Linear (Ridge)- linear regression with standardised inputs.
  Random Forest - bagged decision trees (predicts the change from today).
  XGBoost       - gradient-boosted trees, early stopping on validation (predicts the change).
  LSTM          - recurrent network over the last 14 days of features (predicts the change).
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBRegressor

from .config import N_JOBS, SEED


class Persistence:
    name = "Persistence (baseline)"

    def fit(self, X, y, **_):
        return self

    def predict(self, X):
        return X["aqi"].to_numpy()


class DeltaRegressor(RegressorMixin, BaseEstimator):
    """Learns tomorrow's AQI as today's AQI plus a predicted change.

    Tree ensembles cannot extrapolate beyond the target range seen in training;
    learning the day-to-day change lets them follow high-pollution episodes.
    Requires an `aqi` column (today's AQI) in X.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, eval_set=None, **kw):
        if eval_set is not None:
            kw["eval_set"] = [(Xv, yv - Xv["aqi"]) for Xv, yv in eval_set]
        self.model.fit(X, y - X["aqi"], **kw)
        return self

    def predict(self, X):
        return X["aqi"].to_numpy() + self.model.predict(X)

    @property
    def best_iteration(self):
        return self.model.best_iteration

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def make_rf(**kw):
    params = dict(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=N_JOBS, random_state=SEED)
    params.update(kw)
    return DeltaRegressor(RandomForestRegressor(**params))


def make_xgb(**kw):
    params = dict(
        n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.0, early_stopping_rounds=150, eval_metric="mae",
        random_state=SEED, n_jobs=N_JOBS,
    )
    params.update(kw)
    return DeltaRegressor(XGBRegressor(**params))


# ---------------------------------------------------------------- LSTM

SEQ_LEN = 14


def sequence_index(feats: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Row positions whose previous seq_len-1 rows are the same city on consecutive days."""
    ok = []
    dates = feats["date"].to_numpy()
    cities = feats["city"].to_numpy()
    span = np.timedelta64(seq_len - 1, "D")
    for i in range(seq_len - 1, len(feats)):
        j = i - (seq_len - 1)
        if cities[i] == cities[j] and dates[i] - dates[j] == span:
            ok.append(i)
    return np.asarray(ok, dtype=int)


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMForecaster:
    """Wraps a PyTorch LSTM. Feature tables must be sorted by (city, date)."""

    name = "LSTM"

    def __init__(self, columns, seq_len=SEQ_LEN, epochs=150, patience=15, lr=1e-3, batch=128):
        self.columns = list(columns)
        self.seq_len, self.epochs, self.patience, self.lr, self.batch = seq_len, epochs, patience, lr, batch
        # The network is small enough that CPU training takes about a minute; set AQI_DEVICE=cuda to use a GPU.
        self.device = os.environ.get("AQI_DEVICE", "cpu")

    def _windows(self, feats: pd.DataFrame, rows: np.ndarray) -> torch.Tensor:
        X = self.scaler.transform(feats[self.columns].to_numpy(dtype=float)).astype(np.float32)
        offs = np.arange(-self.seq_len + 1, 1)
        return torch.from_numpy(X[rows[:, None] + offs[None, :]])

    def fit(self, feats: pd.DataFrame, train_rows: np.ndarray, val_rows: np.ndarray):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.scaler = StandardScaler().fit(feats.iloc[train_rows][self.columns].to_numpy(dtype=float))
        # Like DeltaRegressor, the network learns the change from today's AQI.
        y = (feats["target_aqi"] - feats["aqi"]).to_numpy(dtype=np.float32)
        self.y_mean, self.y_std = float(y[train_rows].mean()), float(y[train_rows].std())
        Xtr, Xva = self._windows(feats, train_rows), self._windows(feats, val_rows)
        ytr = torch.from_numpy((y[train_rows] - self.y_mean) / self.y_std)
        yva = y[val_rows]

        self.net = _Net(len(self.columns)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.HuberLoss()
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            for k in range(0, len(perm), self.batch):
                b = perm[k:k + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[b].to(self.device)), ytr[b].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            val_mae = float(np.mean(np.abs(self._predict_windows(Xva) - yva)))
            self.history.append(val_mae)
            if val_mae < best - 1e-3:
                best, bad = val_mae, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history)) + 1
        return self

    def _predict_windows(self, X: torch.Tensor) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = torch.cat([self.net(X[k:k + 1024].to(self.device)).cpu() for k in range(0, len(X), 1024)])
        return out.numpy() * self.y_std + self.y_mean

    def predict(self, feats: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        return feats["aqi"].to_numpy(dtype=float)[rows] + self._predict_windows(self._windows(feats, rows))

    def state(self) -> dict:
        return {
            "columns": self.columns, "seq_len": self.seq_len, "scaler": self.scaler,
            "y_mean": self.y_mean, "y_std": self.y_std,
            "weights": {k: v.cpu() for k, v in self.net.state_dict().items()},
        }

    @classmethod
    def from_state(cls, s: dict) -> "LSTMForecaster":
        m = cls(s["columns"], seq_len=s["seq_len"])
        m.device = "cpu"
        m.scaler, m.y_mean, m.y_std = s["scaler"], s["y_mean"], s["y_std"]
        m.net = _Net(len(m.columns))
        m.net.load_state_dict(s["weights"])
        return m
