# Персональные top-10 рекомендации

**Акопян Микаэла, группа 517**

Датасет: Amazon Reviews 2018 - Video Games (5-core) (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

## Запуск
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
```
## Полученные результаты

| Модель | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|
| SLIM | 0.01630 | 0.02306 |
| CustomALS | 0.01420 | 0.02543 |
| iALS | 0.01375 | 0.02329 |
| ItemKNN | 0.01048 | 0.01546 |
| FM | 0.00825 | 0.01554 |
| Popularity | 0.00747 | 0.01454 |

