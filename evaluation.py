"""Shared full-catalog top-K ranking and macro-averaged metrics."""
import numpy as np


def recommend(model, users, seen, k=10):
    scores = np.asarray(model.score(users), dtype=np.float32)
    if not np.isfinite(scores).all(): raise ValueError('Non-finite model scores')
    for row, user in enumerate(users):
        start, end = seen.indptr[user:user + 2]
        scores[row, seen.indices[start:end]] = -np.inf
    # Stable sorting breaks score ties by the sorted item index.
    ranking = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    if not np.isfinite(np.take_along_axis(scores, ranking, axis=1)).all():
        raise ValueError('Fewer than K unseen candidates')
    return ranking


def evaluate(model, truth, seen, k=10, batch_size=32):
    users = np.array(sorted(truth))
    records = []
    discounts = 1. / np.log2(np.arange(2, k + 2))
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        ranking = recommend(model, batch, seen, k)
        for user, recs in zip(batch, ranking):
            relevant = truth[user]
            hits = np.array([i in relevant for i in recs], dtype=float)
            recall = hits.sum() / len(relevant)
            ideal = discounts[:min(k, len(relevant))].sum()
            records.append((int(user), recall, float(hits @ discounts / ideal)))
    array = np.asarray(records)
    if not len(array): raise ValueError('Empty evaluation cohort')
    return {'Recall@10': float(array[:, 1].mean()), 'NDCG@10': float(array[:, 2].mean()),
            'users': len(users)}, array
