"""Small independent checks for the math and ranking protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.sparse import csr_matrix
from models import ItemKNN, IALS, CustomALS, SLIM
from models.fm import FM
from evaluation import evaluate, recommend

X = csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32))
m = ItemKNN(2).fit(X)
assert np.allclose(m.B.toarray(), (np.ones((3, 3)) - np.eye(3)) * .5)
R = csr_matrix(np.array([[5, 2, 0], [0, 4, 1], [3, 0, 5]], dtype=np.float32))
O = R.copy(); O.data[:] = 1
P = R.copy(); P.data = (P.data >= 4).astype(np.float32)
a = IALS(factors=3, epochs=2).fit(O, P)
b = CustomALS(factors=3, epochs=2).fit(O, R)
assert np.isfinite(a.score([0])).all()
assert not np.allclose(a.score([0]), b.score([0]))
# Verify an ALS update against the original dense normal equation.
Y = np.array([[1., 2.], [2., 0.], [0., 1.]], dtype=np.float32)
solver = IALS(factors=2, reg=2., alpha=3.)
actual = solver._update(O, P, Y)
for u in range(3):
    c = 1 + 3 * O.toarray()[u]
    expected = np.linalg.solve(Y.T @ np.diag(c) @ Y + 2 * np.eye(2), Y.T @ (c * P.toarray()[u]))
    assert np.allclose(actual[u], expected, atol=1e-6)
sl = SLIM(neighbors=2, max_iter=100, l1=.01, l2=1).fit(X)
assert np.all(sl.B.data >= 0) and np.all(sl.B.diagonal() == 0)
# Compare efficient FM scoring with explicit pairwise feature interactions.
fm = FM(2, np.array([[3, 5], [4, 5]]), 6, factors=3)
ids = torch.tensor([1, 3, 5])
v = fm.embedding(ids)
manual = fm.bias + fm.linear(ids).sum()
for i in range(3):
    for j in range(i + 1, 3): manual = manual + (v[i] * v[j]).sum()
assert torch.allclose(fm(torch.tensor([0]), torch.tensor([0]))[0], manual, atol=1e-7)
class Perfect:
    def score(self, users): return np.tile([0., 1., 3., 2.], (len(users), 1))
metrics, _ = evaluate(Perfect(), {0: {2, 3}}, csr_matrix((1, 4)), k=2)
assert metrics['Recall@10'] == 1 and metrics['NDCG@10'] == 1
seen = csr_matrix(([1.], ([0], [2])), shape=(1, 4))
assert 2 not in recommend(Perfect(), [0], seen, k=2)[0]
print('Passed: cosine, ALS normal equations, custom target, SLIM constraints, FM formula, metrics, seen-item exclusion.')
# User metadata must participate in the same second-order FM formula.
fm_user = FM(2, np.array([[3, 5], [4, 5]]), 8, factors=3,
             user_features=np.array([[6], [7]]))
ids = torch.tensor([1, 6, 3, 5])
v = fm_user.embedding(ids)
manual = fm_user.bias + fm_user.linear(ids).sum()
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        manual = manual + (v[a] * v[b]).sum()
assert torch.allclose(fm_user(torch.tensor([0]), torch.tensor([0]))[0], manual, atol=1e-7)
from models.popularity import Popularity
pop = Popularity().fit(X)
assert np.allclose(pop.score([0, 1]), np.tile(np.asarray(X.sum(axis=0)), (2, 1)))
print('Passed: user metadata FM formula and train popularity baseline.')
