# KNRM Search Service - Quora Question Similarity Search

Микросервис для поиска похожих вопросов на основе моделей ранжирования. Система использует комбинированный подход: сначала выполняется быстрый поиск кандидатов с помощью FAISS на основе GloVe эмбеддингов, затем производится точное ранжирование с помощью обученной KNRM модели.

## 🏗️ Архитектура

| Этап | Описание |
|------|----------|
| 1. Запрос | Пользователь отправляет вопрос |
| 2. Фильтрация | LangDetect определяет язык, отсеивает не-английские |
| 3. Векторизация | GloVe преобразует текст в вектор |
| 4. Поиск | FAISS находит 150 наиболее похожих кандидатов |
| 5. Ранжирование | KNRM переранжирует кандидатов по релевантности |
| 6. Результат | Возвращаются топ-10 наиболее релевантных вопросов |

## 🚀 Возможности

- **Фильтрация по языку**: автоматическое определение языка запроса, обработка только английских запросов
- **Векторный поиск**: быстрый поиск кандидатов через FAISS индекс
- **Точное ранжирование**: реранкинг кандидатов с помощью обученной KNRM модели
- **REST API**: удобный интерфейс для интеграциии

## 📦 Установка и запуск

### Требования

- Python 3.9+
- Docker (опционально)
- 4GB RAM минимум

### Локальная установка

```bash
# Клонирование репозитория
git clone https://github.com/popyaske/knrm-search-service.git
cd knrm-search-service

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Запуск сервиса

```bash
# Запуск через Uvicorn (рекомендуется)
python -m uvicorn main:app --host 127.0.0.1 --port 11000 --reload

# Или напрямую
python main.py
```

### Запуск через Docker

```bash
# Сборка образа
docker build -t knrm-search .

# Запуск контейнера
docker run -d --name knrm-search -p 11000:11000 knrm-search

# Просмотр логов
docker logs -f knrm-search
```

### Проверка работы

```bash
# Проверка статуса
curl http://localhost:11000/ping

# Ожидаемый ответ:
# {"status":"ok","message":"Service ready"}

# Или через браузер
open http://localhost:11000/ping
```

## 📋 Структура проекта

knrm-search-service/
├── main.py                   # Основной файл сервиса (FastAPI)
├── knrm_mlp.bin             # Веса MLP части модели
├── requirements.txt          # Зависимости
├── Dockerfile               # Конфигурация Docker
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── project_config.py  # Конфигурация проекта
│   ├── models/
│   │   ├── knrm.py           # Реализация KNRM
│   │   ├── glove_vectorizer.py
│   │   └── searcher.py
│   └── utils/
│       └── preprocessing.py
└── README.md

## 🔧 API Эндпоинты

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/ping` | Проверка готовности сервиса |
| POST | `/update_index` | Обновление FAISS индекса |
| POST | `/query` | Поиск похожих вопросов |

## Примеры запросов


### 1. Проверка статуса
```bash
curl http://localhost:11000/ping
```

### 2. Обновление индекса
```bash
curl -X POST http://localhost:11000/update_index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": {
      "1": "What is machine learning?",
      "2": "How to learn Python?",
      "3": "Best deep learning practices"
    }
  }'
```

### 3. Поиск похожих вопросов
```bash
curl -X POST http://localhost:11000/query \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is ML?",
      "How to code in Python?"
    ]
  }'
```

## 📝 Примечания
- Сервис инициализируется до 120 секунд. /ping вернет status: "ok" только после полной загрузки.

- Индекс строится до 200 секунд через эндпоинт /update_index

- Поиск возвращает до 10 наиболее релевантных вопросов для каждого запроса

- Не-английские запросы фильтруются и возвращают пустой список предложений
