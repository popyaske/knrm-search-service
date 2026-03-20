# Базовый образ с Python 3.9
FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --default-timeout=1000 \
    -r requirements.txt

# Копирование всего проекта
COPY . .

EXPOSE 11000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "11000"]