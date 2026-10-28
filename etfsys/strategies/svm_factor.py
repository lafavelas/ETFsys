"""Plug-in 1: the SVM factor model of Baek et al. (2020), ported to the Strategy interface.

Paper -> plug-in mapping
  Dimension reduction (PCA on 16 indicators)  -> FactorExtractor, fitted on train window only
  Long/short signal detection (zigzag labels) -> zigzag_labels on the label ticker, as targets
  Tuning & validating SVM                     -> GridSearchCV + TimeSeriesSplit (not shuffled k-fold)
  Forecasting module                          -> generate(): predict sign of t+1, map to weights

Fixes vs. the paper: PCA fit on training data only; macro used with a release lag;
time-ordered CV; C tuned alongside gamma; scale-free EMA feature.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..data import MarketData
from ..features import FactorExtractor, ema_ratio, zigzag_labels
from ..strategy import Strategy


class SVMFactorStrategy(Strategy):
    frequency = "M"
    required_data = ("close", "macro")

    def __init__(
        self,
        trade_ticker: str = "SPY",
        label_ticker: str = "SPY",
        n_components: int = 3,
        ema_window: int = 24,
        label_threshold: float = 0.03,
        macro_lag_periods: int = 1,
        gamma_grid: tuple = (0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0),
        c_grid: tuple = (0.1, 1.0, 10.0, 100.0),
        cv_splits: int = 5,
        min_train_periods: int = 48,
        allow_short: bool = False,
        name: str | None = None,
    ):
        self.trade_ticker = trade_ticker
        self.label_ticker = label_ticker
        self.n_components = n_components
        self.ema_window = ema_window
        self.label_threshold = label_threshold
        self.macro_lag_periods = macro_lag_periods
        self.gamma_grid = list(gamma_grid)
        self.c_grid = list(c_grid)
        self.cv_splits = cv_splits
        self.min_train_periods = min_train_periods
        self.allow_short = allow_short
        self.name = name or f"svm_factor_{trade_ticker}"
        self._factors: FactorExtractor | None = None
        self._model = None
        self._diag: dict = {}

    # ----------------------------------------------------------------- helpers
    def _monthly(self, data: MarketData) -> MarketData:
        return data.resample(self.frequency)

    def _macro(self, m: MarketData) -> pd.DataFrame:
        # Point-in-time guard: a month's indicators are usable only after release.
        return m.macro.shift(self.macro_lag_periods)

    def _features(self, m: MarketData) -> pd.DataFrame:
        F = self._factors.transform(self._macro(m).dropna())
        F["ema_ratio"] = ema_ratio(m.close[self.trade_ticker], self.ema_window)
        return F.dropna()

    # --------------------------------------------------------------- interface
    def fit(self, data: MarketData, train_end: pd.Timestamp) -> None:
        self.check_data(data)
        m = self._monthly(data.truncate(train_end))
        macro = self._macro(m).dropna()
        if len(macro) < self.min_train_periods:
            self._model = None
            return
        self._factors = FactorExtractor(self.n_components).fit(macro)
        X = self._features(m)
        y = zigzag_labels(m.close[self.label_ticker], self.label_threshold).shift(
            -1
        )  # target = label at t+1
        df = X.join(y.rename("y")).dropna()
        X_train, y_train = df.drop(columns="y"), df["y"].astype(int)

        if y_train.nunique() < 2:
            self._model = ("constant", int(y_train.iloc[0]))
            return
        pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="rbf"))])
        search = GridSearchCV(
            pipe,
            {"svc__gamma": self.gamma_grid, "svc__C": self.c_grid},
            cv=TimeSeriesSplit(n_splits=self.cv_splits),
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        with warnings.catch_warnings():
            # Early time-series folds can hold one class only; their scores are noisy, not wrong.
            warnings.simplefilter("ignore", UserWarning)
            search.fit(X_train, y_train)
        self._model = search.best_estimator_
        cv = pd.DataFrame(search.cv_results_)
        self._diag = {
            "train_end": str(pd.Timestamp(train_end).date()),
            "n_train": len(y_train),
            "share_up_labels": round(float((y_train == 1).mean()), 3),
            "best_params": {k: float(v) for k, v in search.best_params_.items()},
            "cv_balanced_accuracy": round(float(search.best_score_), 3),
            "cv_std_of_best": round(float(cv.loc[search.best_index_, "std_test_score"]), 3),
            "explained_variance": self._factors.explained_variance().round(3).to_dict(),
        }

    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)  # not enough history yet: stay in cash
        if isinstance(self._model, tuple):
            pred = self._model[1]
        else:
            X = self._features(self._monthly(data))
            if X.empty:
                return pd.Series(dtype=float)
            pred = int(self._model.predict(X.iloc[[-1]])[0])
        if pred < 0 and not self.allow_short:
            return pd.Series({self.trade_ticker: 0.0})
        return pd.Series({self.trade_ticker: float(np.sign(pred))})

    def diagnostics(self) -> dict:
        return dict(self._diag)

    def loadings(self) -> pd.DataFrame | None:
        return None if self._factors is None else self._factors.loadings()
