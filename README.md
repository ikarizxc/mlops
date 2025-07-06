# MLOps Project - Home Credit Default Risk

Этот проект представляет собой MLOps инфраструктуру для решения задачи предсказания дефолта по кредитам Home Credit. Проект включает в себя автоматизированный пайплайн обработки данных, обучения моделей и их отслеживания с использованием Apache Airflow, MLflow и PostgreSQL.

## 🏗️ Архитектура проекта

Проект состоит из следующих компонентов:

- **Apache Airflow** - оркестрация пайплайнов данных и обучения моделей
- **PostgreSQL** - хранение данных и метаданных
- **Redis** - брокер сообщений для Celery
- **MLflow** - отслеживание экспериментов и версионирование моделей
- **Docker** - контейнеризация всех сервисов

### Дополнительные требования
- **Kaggle API**: для загрузки данных
- **Python 3.12**: для локальной разработки (опционально)

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd mlops
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корневой директории проекта:

```bash
# Airflow
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=database
POSTGRES_PORT=5432

# MLflow
MLFLOW_SERVICE_PORT=5000

# AWS S3 (для MLflow artifacts)
MLFLOW_S3_ENDPOINT_URL=your_s3_endpoint
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
BUCKET_NAME=your_bucket_name
```

### 3. Настройка Kaggle API

Для загрузки данных необходимо настроить Kaggle API:

1. Зарегистрируйтесь на [Kaggle](https://www.kaggle.com/)
2. Перейдите в настройки аккаунта → API → Create New API Token
3. Скачайте файл `kaggle.json`
4. Создайте директорию и поместите туда файл:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Загрузка данных

```bash
# Сделайте скрипт исполняемым
chmod +x download_data.sh

# Загрузите данные с Kaggle
./download_data.sh
```

### 5. Запуск инфраструктуры

```bash
# Сборка и запуск всех сервисов
docker-compose up -d --build

# Проверка статуса сервисов
docker-compose ps
```

### 6. Загрузка данных в PostgreSQL

```bash
# Загрузка данных в базу
python load_data.py --user postgres --password postgres --database database --address localhost --port 5432
```

### 7. Настройка подключений в Airflow

После запуска Airflow перейдите в веб-интерфейс по адресу `http://localhost:8080` и настройте подключение к PostgreSQL:

1. Войдите с учетными данными: `admin/admin`
2. Перейдите в Admin → Connections
3. Создайте новое подключение для базы данных с данными для обучения:
   - **Connection Id**: `home-credit-default-risk`
   - **Connection Type**: `Postgres`
   - **Host**: `postgres`
   - **Schema**: `database`
   - **Login**: `postgres`
   - **Password**: `postgres`
   - **Port**: `5432`

## 📊 Доступные сервисы

После успешного запуска будут доступны следующие сервисы:

| Сервис | URL | Описание |
|--------|-----|----------|
| **Airflow Web UI** | http://localhost:8080 | Веб-интерфейс для управления DAG'ами |
| **Airflow API** | http://localhost:8080/api/v1 | REST API для Airflow |
| **MLflow UI** | http://localhost:5000 | Интерфейс для отслеживания экспериментов |
| **PostgreSQL** | localhost:5432 | База данных |
| **Flower** | http://localhost:5555 | Мониторинг Celery задач (опционально) |

## 🔄 DAG'и (Directed Acyclic Graphs)

Проект содержит два основных DAG'а:

### 1. `data_preparation` DAG
Выполняет подготовку данных для обучения модели:
- **Проверка подключения к БД** - проверяет доступность PostgreSQL
- **Извлечение сырых данных** - загружает данные из таблицы `application_train`
- **Трансформация данных** - выполняет предобработку (обработка выбросов, заполнение пропусков, создание новых признаков)
- **Загрузка в таблицу** - сохраняет обработанные данные в таблицу `train_data`
- **Валидация данных** - проверяет целостность загруженных данных

### 2. `model_training` DAG
Обучает и оценивает модель машинного обучения:
- **Получение данных** - загружает подготовленные данные
- **Проверка качества** - оценивает достаточность данных для обучения
- **Обучение модели** - обучает CatBoost классификатор
- **Логирование метрик** - сохраняет метрики в MLflow
- **Сохранение артефактов** - сохраняет графики и модель
- **Регистрация модели** - регистрирует лучшую модель в MLflow Model Registry

## 🛠️ Разработка и отладка

### Локальная разработка

Для локальной разработки можно использовать виртуальное окружение:

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Отладка DAG'ов

```bash
# Запуск Airflow CLI для отладки
docker-compose run --rm airflow-cli airflow dags test data_preparation 2024-01-01

# Просмотр логов
docker-compose logs airflow-scheduler
docker-compose logs airflow-worker
```

## 📁 Структура проекта

```
mlops/
├── airflow/
│   ├── config/                 # Конфигурация Airflow
│   ├── dags/                   # DAG'и для оркестрации
│   ├── docker/                 # Docker конфигурации
│   ├── plugins/                # Плагины Airflow
│   └── scripts/                # Скрипты обработки данных
│       ├── data_preparation/   # Скрипты подготовки данных
│       └── preprocessors/      # Препроцессоры данных
├── docker-compose.yaml         # Конфигурация Docker Compose
├── Dockerfile                  # Основной Docker образ
├── requirements.txt            # Python зависимости
├── download_data.sh           # Скрипт загрузки данных
├── load_data.py               # Скрипт загрузки в БД
└── README.md                  # Документация
```

## 🔧 Конфигурация

### Настройка Airflow

Основные настройки Airflow находятся в файле `airflow/config/airflow.cfg`. Ключевые параметры:

- `executor = CeleryExecutor` - использование Celery для распределенного выполнения
- `load_examples = False` - отключение примеров DAG'ов
- `dags_are_paused_at_creation = True` - DAG'и создаются в приостановленном состоянии

### Настройка MLflow

MLflow настроен для работы с S3-совместимым хранилищем. Для локальной разработки можно использовать MinIO или настроить локальное хранилище.

## 📈 Мониторинг и логирование

### Логи Airflow

```bash
# Просмотр логов планировщика
docker-compose logs -f airflow-scheduler

# Просмотр логов воркера
docker-compose logs -f airflow-worker

# Просмотр логов конкретного DAG
docker-compose exec airflow-scheduler airflow dags backfill data_preparation
```

### Метрики MLflow

- Отслеживание экспериментов: http://localhost:5000
- Регистр моделей: http://localhost:5000/#/models
- Сравнение запусков: http://localhost:5000/#/experiments

## 📚 Дополнительные ресурсы

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk)
