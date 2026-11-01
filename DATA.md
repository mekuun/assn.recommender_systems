# Amazon Reviews 2018 — Video Games

Источник: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Загружаем опубликованную авторами 5-core выборку отзывов и полные метаданные категории Video Games. Категория включает игры, консоли и аксессуары.

Запуск из папки задания: `python3 scripts/download_data.py` (Python 3, без дополнительных зависимостей).

Файлы сохраняются в `data/raw/`. URL, размеры и SHA-256 записываются в `data/download_manifest.json`. Данные исключены из Git через `.gitignore`.

Основные поля отзывов: `reviewerID` — пользователь, `asin` — товар, `overall` — оценка 1–5, `unixReviewTime` — время отзыва. Метаданные соединяются по `asin`; содержат название, категории, бренд и описание. Публикация метаданных не гарантирует их историческую доступность: агрегированные рейтинги, ранги продаж и связи also_buy/also_view не следует использовать как признаки без проверки утечки.

5-core сформирован авторами по всему периоду. Это ограничение отбора нужно описать при временной оценке. Очистка, временное разбиение и ограничения описаны в `experiments.ipynb`.

Ссылка для цитирования: Jianmo Ni, Jiacheng Li, Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. EMNLP, 2019.
