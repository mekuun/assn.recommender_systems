"""Train-only nonpersonalized popularity baseline."""
import numpy as np

class Popularity:
    def fit(self, X):
        self.popularity = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.popularity, (len(users), 1))
