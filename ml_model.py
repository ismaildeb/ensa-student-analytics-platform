from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class IsotonicCalibratedModel(ClassifierMixin, BaseEstimator):
    """
    Pickle-friendly calibrated model:
    - base: any classifier pipeline with predict_proba
    - calibrator: fitted IsotonicRegression that maps base proba -> calibrated proba
    """

    def __init__(self, base, calibrator: IsotonicRegression):
        self.base = base
        self.calibrator = calibrator

    def fit(self, X, y=None):
        # No-op: the underlying components are fitted externally.
        return self

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        pc = self.calibrator.predict(p)
        pc = np.clip(pc, 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
