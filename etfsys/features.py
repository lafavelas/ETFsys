"""Shared feature and label library.

Everything here is a pure function or a fit/transform object, so any strategy can
reuse it. The paper's ML Analytic Module (PCA factors, breakout labels, EMA) lives
here instead of inside one model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------- factors
class FactorExtractor:
    """Standardize indicators, then PCA (Baek et al. 2020, eqs. 1-3).

    Fit on the TRAINING window only, then transform any later date with the same
    scaler and loadings. PCA signs are arbitrary, so each component is flipped to
    make its largest absolute loading positive; refits then keep a stable meaning.
    """

    def __init__(self, n_components: int = 3, names: tuple[str, ...] | None = None):
        self.n_components = n_components
        self.names = names or tuple(f"factor_{i + 1}" for i in range(n_components))
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._signs: np.ndarray | None = None
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "FactorExtractor":
        self.columns = list(X.columns)
        Z = self.scaler.fit_transform(X.values)
        self.pca.fit(Z)
        comps = self.pca.components_
        biggest = np.abs(comps).argmax(axis=1)
        self._signs = np.sign(comps[np.arange(len(comps)), biggest])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.scaler.transform(X[self.columns].values)
        F = self.pca.transform(Z) * self._signs
        return pd.DataFrame(F, index=X.index, columns=list(self.names))

    def loadings(self) -> pd.DataFrame:
        """Indicator x factor loadings, the paper's Table 2 panel A."""
        L = (self.pca.components_ * self._signs[:, None]).T
        return pd.DataFrame(L, index=self.columns, columns=list(self.names))

    def explained_variance(self) -> pd.Series:
        return pd.Series(self.pca.explained_variance_ratio_, index=list(self.names))


# ---------------------------------------------------------------------- labels
def zigzag_labels(close: pd.Series, threshold: float = 0.03) -> pd.Series:
    """+1 / -1 per period: is the move INTO this period part of an up or down segment?

    Segments are moves of at least `threshold` between confirmed pivots (Baek et al.
    2020, section 3.3). Labels look ahead by construction (a pivot is only known after
    the reversal), so use them only as TARGETS, never as features. The last segment
    is unconfirmed and labelled by its direction so far.
    """
    p = close.to_numpy(dtype=float)
    n = len(p)
    labels = np.zeros(n)
    if n < 2:
        return pd.Series(labels, index=close.index)
    direction, pivot, ext = 0, 0, 0
    for i in range(1, n):
        if direction == 0:
            chg = p[i] / p[pivot] - 1
            if chg >= threshold:
                direction, ext = 1, i
            elif chg <= -threshold:
                direction, ext = -1, i
            continue
        if direction * (p[i] - p[ext]) >= 0:  # trend extends
            ext = i
        elif direction * (p[i] / p[ext] - 1) <= -threshold:  # confirmed reversal
            labels[pivot + 1 : ext + 1] = direction
            pivot, ext, direction = ext, i, -direction
    if direction == 0:
        direction = 1 if p[-1] >= p[0] else -1
    labels[pivot + 1 :] = direction
    labels[0] = labels[1]
    return pd.Series(labels, index=close.index, name="label")


def breakout_signals(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 12) -> pd.Series:
    """+1 if close breaks above the prior n-period high, -1 below the prior low, else 0."""
    prior_high = high.shift(1).rolling(n - 1, min_periods=n - 1).max()
    prior_low = low.shift(1).rolling(n - 1, min_periods=n - 1).min()
    sig = pd.Series(0.0, index=close.index)
    sig[close > prior_high] = 1.0
    sig[close < prior_low] = -1.0
    return sig


# --------------------------------------------------------------- price features
def ema_ratio(close: pd.Series, window: int = 24) -> pd.Series:
    """close / EMA - 1: scale-free trend feature (raw EMA levels are bad SVM inputs)."""
    return close / close.ewm(span=window, adjust=False).mean() - 1


def momentum_score(close: pd.DataFrame, horizons=(4, 12, 26)) -> pd.DataFrame:
    """Average of trailing returns over several horizons (in periods of `close`)."""
    parts = [close.pct_change(h, fill_method=None) for h in horizons]
    return sum(parts) / len(parts)


def realized_vol(returns: pd.DataFrame, window: int = 63, periods_per_year: int = 252) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(periods_per_year)
