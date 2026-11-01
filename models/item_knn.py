"""ItemKNN: cosine similarity and positive-history weighted sum."""
import numpy as np
from scipy.sparse import csr_matrix, diags


def top_neighbors(matrix, k):
    matrix = matrix.tocsr()
    indices, values, indptr = [], [], [0]
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i:i + 2]
        ix, val = matrix.indices[start:end], matrix.data[start:end]
        if len(val) > k:
            chosen = np.argpartition(val, -k)[-k:]
            ix, val = ix[chosen], val[chosen]
        order = np.lexsort((ix, -val))
        indices.extend(ix[order]); values.extend(val[order]); indptr.append(len(indices))
    return csr_matrix((np.asarray(values, dtype=np.float32), indices, indptr), shape=matrix.shape)


def cosine_matrix(X):
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel())
    inv = np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)
    normalized = X @ diags(inv)
    similarity = (normalized.T @ normalized).tocsr()
    coo = similarity.tocoo()
    keep = coo.row != coo.col
    similarity = csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape)
    similarity.eliminate_zeros()
    return similarity


class ItemKNN:
    def __init__(self, k=50):
        self.k = k

    def fit(self, X):
        self.X = X.tocsr()
        # Each target column retains its k nearest source items.
        self.B = top_neighbors(cosine_matrix(X), self.k).T.tocsr()
        return self

    def score(self, users):
        return (self.X[users] @ self.B).toarray()
