"""Second-order FM with sparse user/item/brand/category features, trained with BCE."""
import numpy as np
import torch
from torch import nn


class FM(nn.Module):
    def __init__(self, n_users, item_features, n_features, factors=16, lr=.01,
                 reg=1e-5, epochs=4, seed=42, batch_size=4096, user_features=None):
        super().__init__()
        torch.manual_seed(seed)
        self.n_users, self.n_items = n_users, len(item_features)
        self.register_buffer('item_features', torch.as_tensor(item_features, dtype=torch.long))
        if user_features is None:
            user_features = np.empty((n_users, 0), dtype=np.int64)
        self.register_buffer('user_features', torch.as_tensor(user_features, dtype=torch.long))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.embedding = nn.Embedding(n_features, factors, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=.03)
        with torch.no_grad(): self.embedding.weight[0].zero_()
        self.lr, self.reg, self.epochs = lr, reg, epochs
        self.seed, self.batch_size = seed, batch_size

    def forward(self, users, items):
        ids = torch.cat((users[:, None] + 1, self.user_features[users], self.item_features[items]), dim=1)
        vectors = self.embedding(ids)
        interactions = .5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.linear(ids).sum(dim=(1, 2)) + interactions

    def fit(self, users, items, labels, observed):
        rng = np.random.default_rng(self.seed)
        # One unobserved negative per observed positive; reject ALL train-rated items.
        positive_users = users[np.asarray(labels) == 1]
        negative_items = rng.integers(self.n_items, size=len(positive_users))
        while True:
            bad = np.asarray(observed[positive_users, negative_items]).ravel() > 0
            if not bad.any(): break
            negative_items[bad] = rng.integers(self.n_items, size=int(bad.sum()))
        u = torch.as_tensor(np.concatenate((users, positive_users)), dtype=torch.long)
        i = torch.as_tensor(np.concatenate((items, negative_items)), dtype=torch.long)
        y = torch.as_tensor(np.concatenate((labels, np.zeros(len(positive_users)))), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(y)); total = 0.
            for start in range(0, len(y), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = self(u[batch], i[batch])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
                loss.backward(); optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history.append(total / len(y))
        return self

    def score(self, users):
        result = np.empty((len(users), self.n_items), dtype=np.float32)
        with torch.no_grad():
            # Bounded feature tensors even for a full-catalog evaluation.
            for row, user in enumerate(users):
                for start in range(0, self.n_items, 4096):
                    items = torch.arange(start, min(start + 4096, self.n_items))
                    us = torch.full_like(items, int(user))
                    result[row, start:start + len(items)] = self(us, items).numpy()
        return result
