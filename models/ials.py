"""Weighted implicit ALS with explicit preference and confidence matrices."""
import numpy as np


class IALS:
    def __init__(self, factors=16, reg=10., alpha=10., epochs=5, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.epochs, self.seed = epochs, seed

    def fit(self, observed, preference):
        # observed has ones for ALL ratings, preference can have zero values there.
        observed = observed.tocsr(); preference = preference.tocsr()
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, .05, (observed.shape[0], self.factors)).astype(np.float32)
        self.V = rng.normal(0, .05, (observed.shape[1], self.factors)).astype(np.float32)
        for _ in range(self.epochs):
            self.U = self._update(observed, preference, self.V)
            self.V = self._update(observed.T.tocsr(), preference.T.tocsr(), self.U)
        return self

    def _update(self, observed, preference, fixed):
        base = fixed.T @ fixed + self.reg * np.eye(self.factors, dtype=np.float32)
        out = np.zeros((observed.shape[0], self.factors), dtype=np.float32)
        for row in range(observed.shape[0]):
            a, b = observed.indptr[row:row + 2]
            ix = observed.indices[a:b]
            if not len(ix): continue
            local = fixed[ix]
            # Dense lookup only for a user's observed items, not the full catalogue.
            pa, pb = preference.indptr[row:row + 2]
            if not np.array_equal(preference.indices[pa:pb], ix):
                raise ValueError("Preference must store every observed entry, including zeros")
            target = preference.data[pa:pb]
            A = base + self.alpha * (local.T @ local)
            rhs = (1. + self.alpha) * (local.T @ target)
            out[row] = np.linalg.solve(A, rhs)
        return out

    def score(self, users):
        return self.U[users] @ self.V.T
