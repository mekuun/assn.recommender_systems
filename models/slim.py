"""Nonnegative SLIM, coordinate descent on fixed train-only candidate supports."""
import numpy as np
from scipy.sparse import csr_matrix
from .item_knn import cosine_matrix, top_neighbors


class SLIM:
    def __init__(self, l1=5., l2=10., neighbors=60, max_iter=60, tol=1e-4):
        self.l1, self.l2, self.neighbors = l1, l2, neighbors
        self.max_iter, self.tol = max_iter, tol

    def fit(self, X):
        self.X = X.tocsr()
        G = (X.T @ X).tocsr().astype(np.float64)
        support = top_neighbors(cosine_matrix(X), self.neighbors)
        rows, cols, vals = [], [], []
        converged = 0
        for j in range(X.shape[1]):
            s, e = support.indptr[j:j + 2]
            ix = support.indices[s:e]
            if not len(ix):
                converged += 1; continue
            H = G[ix][:, ix].toarray()
            H.flat[::len(ix) + 1] += self.l2
            b = G[ix, j].toarray().ravel() - self.l1
            w = np.zeros(len(ix)); residual = b.copy()
            for iteration in range(self.max_iter):
                change = 0.
                for k in range(len(ix)):
                    new = max(0., w[k] + residual[k] / H[k, k])
                    delta = new - w[k]
                    if delta:
                        residual -= H[:, k] * delta
                        w[k] = new
                        change = max(change, abs(delta))
                if change < self.tol:
                    converged += 1; break
            keep = w > 0
            rows.extend(ix[keep]); cols.extend([j] * int(keep.sum())); vals.extend(w[keep])
        self.B = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=G.shape)
        self.converged_fraction = converged / X.shape[1]
        return self

    def score(self, users):
        return (self.X[users] @ self.B).toarray()
