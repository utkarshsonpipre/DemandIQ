"""PyTorch LSTM: sequence-to-sequence, last `seq_len` sales -> next `horizon`.
One global model trained across all series with per-series min-max scaling."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.constants import DATE_COL, SERIES_COL, TARGET_COL
from src.logger import get_logger
from src.models.base_model import BaseModel

logger = get_logger(__name__)


class _Seq2Seq(nn.Module):
    def __init__(self, units: list[int], horizon: int, dropout: float):
        super().__init__()
        layers = []
        in_size = 1
        for u in units:
            layers.append(nn.LSTM(in_size, u, batch_first=True))
            in_size = u
        self.lstms = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(units[-1], horizon)

    def forward(self, x):  # x: (batch, seq_len, 1)
        for lstm in self.lstms:
            x, _ = lstm(x)
            x = self.dropout(x)
        return self.head(x[:, -1, :])  # (batch, horizon)


class LSTMModel(BaseModel):
    name = "LSTM"

    def __init__(self, params: dict, horizon: int = 30, seed: int = 42):
        self.seq_len = params.get("input_sequence_length", 60)
        self.units = params.get("lstm_units", [64, 32])
        self.dropout = params.get("dropout", 0.2)
        self.batch_size = params.get("batch_size", 32)
        self.epochs = params.get("epochs", 30)
        self.lr = params.get("learning_rate", 0.001)
        self.horizon = horizon
        self.seed = seed
        self.scalers: dict[str, tuple[float, float]] = {}  # series -> (min, max)
        self.net: _Seq2Seq | None = None

    def _scale(self, sid: str, y: np.ndarray) -> np.ndarray:
        lo, hi = self.scalers[sid]
        return (y - lo) / (hi - lo + 1e-9)

    def _unscale(self, sid: str, y: np.ndarray) -> np.ndarray:
        lo, hi = self.scalers[sid]
        return y * (hi - lo + 1e-9) + lo

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        torch.manual_seed(self.seed)
        X, Y = [], []
        for sid, g in train.groupby(SERIES_COL):
            y = g.sort_values(DATE_COL)[TARGET_COL].astype(float).to_numpy()
            self.scalers[sid] = (float(y.min()), float(y.max()))
            ys = self._scale(sid, y)
            for i in range(len(ys) - self.seq_len - self.horizon + 1):
                X.append(ys[i:i + self.seq_len])
                Y.append(ys[i + self.seq_len:i + self.seq_len + self.horizon])
        if not X:
            raise RuntimeError("Not enough history to build LSTM sequences")
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)

        self.net = _Seq2Seq(self.units, self.horizon, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(X, Y)
        n_val = max(1, int(0.1 * len(ds)))
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val], torch.Generator().manual_seed(self.seed))
        dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        Xv, Yv = val_ds[:]

        best_val, best_state, patience = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(Xv), Yv).item()
            if val_loss < best_val - 1e-5:
                best_val, best_state, patience = val_loss, self.net.state_dict(), 0
            else:
                patience += 1
                if patience >= 5:
                    logger.info("LSTM early stop at epoch %d (val %.5f)", epoch, best_val)
                    break
        if best_state:
            self.net.load_state_dict(best_state)

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        if history is None:
            raise ValueError("LSTM predict requires history")
        self.net.eval()
        preds = np.zeros(len(future))
        pos = future.reset_index(drop=True)
        for sid, g in pos.groupby(SERIES_COL):
            h = history[history[SERIES_COL] == sid].sort_values(DATE_COL)
            y = h[TARGET_COL].astype(float).to_numpy()
            if sid not in self.scalers:  # unseen series: scale by its own history
                self.scalers[sid] = (float(y.min()), float(y.max()))
            seq = list(self._scale(sid, y[-self.seq_len:]))
            out: list[float] = []
            while len(out) < len(g):  # roll forward in horizon-sized chunks
                x = torch.tensor(seq[-self.seq_len:], dtype=torch.float32).reshape(1, -1, 1)
                with torch.no_grad():
                    chunk = self.net(x).numpy().ravel()
                out.extend(chunk)
                seq.extend(chunk)
            preds[g.index] = np.clip(self._unscale(sid, np.array(out[:len(g)])), 0, None)
        return preds
