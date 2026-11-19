# Sports Physiology RAG — API для двух книг

RAG-система над:
1. **Уилмор и Костилл — Физиология спорта и двигательной активности** (459 стр., 1141 чанк)
2. **Иссурин — Подготовка спортсменов XXI века** (460 стр., 1347 чанков)

## Возможности

- **Лексический поиск (FTS5)** — работает без ML-зависимостей, быстрый
- **Векторный поиск (LSA 128D)** — требует `numpy` + `joblib` (опционально)
- **Гибридный режим** — комбинирует FTS + векторы с реранжированием
- **Генерация через Nemotron-3-Ultra** (NVIDIA API) с цитированием PDF-страниц
- **FastAPI** — REST API, деплой на Railway (бесплатно)

## Локальный запуск

```bash
# Установить зависимости (для FTS достаточно sqlite3 из stdlib)
pip install -r requirements.txt

# Для векторного поиска дополнительно:
pip install numpy scikit-learn joblib

# Запуск сервера
export NVIDIA_API_KEY=ваш_ключ
python server.py
# Открыть http://localhost:8000/docs
```

## API

### POST /query
```json
{
  "query": "как повысить лактатный порог",
  "top_k": 5,
  "chapter": 5,
  "mode": "hybrid"
}
```

**mode**: `fts` | `vector` | `hybrid` (default)

### Response
```json
{
  "query": "как повысить лактатный порог",
  "answer": "Согласно Уилмору и Костиллу [Источник 1, PDF-с. 90], лактатный порог... [Источник 2, PDF-с. 154] Иссурин рекомендует...",
  "sources": [
    {"skill": "physiology-sport-skill", "book": "...", "chapter": 5, "pdf_page": 90, "part": "2", "score": 0.016},
    {"skill": "issurin-training-skill", "book": "...", "chapter": 6, "pdf_page": 154, "part": "1", "score": 0.014}
  ]
}
```

## Деплой на Railway (бесплатно)

1. Создайте GitHub репозиторий с этой папкой
2. Railway → New Project → Deploy from GitHub
3. Добавьте переменную окружения: `NVIDIA_API_KEY`
4. Railway сам соберёт Dockerfile и выдаст публичный URL

## Архитектура

```
query → MultiBookRAG.search_all()
    → BookSearcher (FTS5) для каждой книги
    → BookSearcher (LSA vectors) для каждой книги (опционально)
    → fused score ranking
    → format_context()
    → build_prompt() с цитированием [Источник N]
    → Nemotron-3-Ultra (NVIDIA API)
    → answer + sources
```

## Файлы данных

Скопированы из Hermes skills:
- `data/physiology-sport-skill/` — SQLite FTS5 + LSA индексы
- `data/issurin-training-skill/` — SQLite FTS5 + LSA индексы

Оригинальные PDF: `references/source.pdf` внутри каждого навыка в Hermes.

## Примеры запросов

- `лактатный порог и выносливость` — физиология + методика
- `блоковая периодизация` — Иссурин гл. 6
- `адаптация сердечно-сосудистой системы` — Уилмор/Костилл гл. 10
- `перенос тренированности` — Иссурин гл. 4
- `поляризованная тренировка выносливости` — обе книги