from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HGB_AVAILABLE = True
except Exception:
    HGB_AVAILABLE = False


class BootstrappedRF:
    def __init__(
        self,
        n_models: int = 20,
        seed: int = 0,
        n_estimators: int = 600,
        n_jobs: int = -1,
    ):
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)
        self.models: list[RandomForestRegressor] = []
        self.fitted = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.models = []
        n = x_train.shape[0]
        for _ in range(self.n_models):
            idx = self.rng.integers(0, n, size=n)
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=2,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                n_jobs=self.n_jobs,
            )
            model.fit(x_train[idx], y_train[idx])
            self.models.append(model)
        self.fitted = True

    def predict_mean_std(self, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("RF ensemble not fitted.")
        preds = np.stack([m.predict(x_eval) for m in self.models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


class MonotoneHGBWrapper:
    def __init__(
        self,
        monotonic_cst: list[int],
        seed: int = 0,
        max_depth: int = 6,
        learning_rate: float = 0.07,
        max_iter: int = 600,
        min_samples_leaf: int = 20,
    ):
        if not HGB_AVAILABLE:
            raise RuntimeError("HistGradientBoostingRegressor not available.")
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            monotonic_cst=monotonic_cst,
        )
        self.fitted = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)
        self.fitted = True

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Monotone model not fitted.")
        return self.model.predict(x_eval)
