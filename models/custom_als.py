"""Rating-aware target; confidence is identical to the baseline IALS."""
from .ials import IALS


class CustomALS(IALS):
    def fit(self, observed, ratings):
        target = ratings.copy().astype('float32')
        target.data = (target.data - 3.) / 2.
        return super().fit(observed, target)
