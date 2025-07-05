from datetime import timedelta
import os
import logging

from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

import mlflow
import mlflow.catboost
import mlflow.experiments
from mlflow.models import infer_signature

from matplotlib import pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score, 
    roc_auc_score, 
    f1_score,
    roc_curve
)

from pendulum import today

from scripts.preprocessors.base_preprocessor import BasePreprocessor

_logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

args = {
    "owner": "Sviridyuk Sergey",
    "email": ["sergey.sviridyuk.41@gmail.com"],
    'email_on_failure': False,
}

dag = DAG(
    dag_id="model_training",
    default_args=args,
    max_active_runs=1,
    max_active_tasks=3,
    schedule="@daily",
    start_date=today("UTC").subtract(days=1),
    tags=["data", "models", "training", "python", "mlflow"],
)

def get_train_data(**kwargs) -> str:
    """
    Получает обучающие данные из Postgres и сохраняет их во временный parquet-файл.

    Аргументы:
        **kwargs: аргументы, передаваемые Airflow (не используются в функции).

    Возвращает:
        str: путь к сохранённому parquet-файлу с обучающими данными.
    """
    table_name = 'train_data'
    postgres_hook = PostgresHook(postgres_conn_id='home-credit-default-risk')
    sql = f"SELECT * FROM {table_name};"
    df: pd.DataFrame = postgres_hook.get_pandas_df(sql=sql)
    _logger.info(f"Загружено {len(df)} строк из таблицы '{table_name}'")
    
    temp_directory = 'opt/airflow/tmp'
    os.makedirs(temp_directory, exist_ok=True)
    
    temp_path = f"{temp_directory}/{table_name}.parquet"
    df.to_parquet(temp_path, index=False)
    _logger.info(f"Данные для обучения временно сохранены в '{temp_path}'")
    return temp_path

def check_data_quality(**kwargs) -> str:
    """
    Проверяет качество загруженных данных и выбирает следующий шаг в DAG.

    Аргументы:
        **kwargs: аргументы, передаваемые Airflow, содержит объект ti для XCom.

    Возвращает:
        str: task_id следующей задачи ('notify_bad_data' при недостаточном объёме данных,
             иначе 'train_model').
    """
    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids="get_train_data")
    
    df = pd.read_parquet(path)
    
    if len(df) < 100_000:
        return 'notify_bad_data'
    
    return 'train_model'

def train_model(**kwargs) -> dict:
    """
    Обучает CatBoostClassifier на подготовленных данных, оценивает модель и логирует результаты в MLflow.

    Аргументы:
        **kwargs: аргументы Airflow, содержит объект ti для XCom.

    Возвращает:
        dict: информация о запуске MLflow, включая run_id, метрики и важности признаков.
    """
    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids="get_train_data")
    
    df = pd.read_parquet(path)
    
    X, y = df.drop(columns=['TARGET']), df['TARGET']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=47)
    
    base_preprocessor = BasePreprocessor()
    cat_features = base_preprocessor.get_categorical_features(X_train)
    
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)

    classifier = CatBoostClassifier()
    classifier.fit(train_pool)
    
    preds = classifier.predict(val_pool)
    proba = classifier.predict_proba(val_pool)[:, 1]
    
    acc   = accuracy_score(y_val, preds)
    prec  = precision_score(y_val, preds)
    rec   = recall_score(y_val, preds)
    f1    = f1_score(y_val, preds)
    auc   = roc_auc_score(y_val, proba)
    fpr, tpr, _ = roc_curve(y_val, proba)

    feat_imps = classifier.get_feature_importance()
    feat_names= classifier.feature_names_

    sig = infer_signature(X_val, preds)

    out = {
      'metrics': {
         'accuracy':  acc,
         'precision': prec,
         'recall':    rec,
         'f1':        f1,
         'roc_auc':   auc,
         'fpr':       fpr.tolist(),
         'tpr':       tpr.tolist()
      },
      'feature_importances': {
         'names': feat_names,
         'values': feat_imps.tolist()
      }
    }
    
    exp_name = 'home-credit-default-risk'
    exp = mlflow.get_experiment_by_name(exp_name)

    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id
    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        mlflow.catboost.log_model(
            classifier, 
            "catboost", 
            signature=sig,
            run_id=run_id)
    
    out['run_id'] = run_id
    
    return out

def log_metrics(**kwargs) -> None:
    """
    Извлекает метрики из XCom и логирует их в соответствующий запуск MLflow.

    Аргументы:
        **kwargs: аргументы Airflow, содержит объект ti для XCom.
    """
    ti = kwargs['ti']
    run_info = ti.xcom_pull(task_ids="train_model")
    run_id = run_info['run_id']
    metrics = run_info['metrics']
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(
        run_id=run_id,
        metrics=scalar_metrics
    )

def log_artifacts(**kwargs):
    """
    Сохраняет и логирует артефакты модели в MLflow:
      1) График важности признаков
      2) ROC-кривую на валидационном наборе

    Аргументы:
        **kwargs: аргументы Airflow, содержит объект ti для XCom.
    """
    ti = kwargs['ti']
    run_info = ti.xcom_pull(task_ids="train_model")
    metrics = run_info['metrics']
    feature_importances = run_info['feature_importances']

    # Создаём директорию для артефактов, если её нет
    artifacts_dir = '/opt/airflow/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    # Важность признаков
    fi = feature_importances['values']
    names = feature_importances['names']
    fig, ax = plt.subplots(figsize=(6, max(4, len(fi) * 0.3)))
    ax.barh(names, fi)
    ax.set_title("Важность признаков")
    fi_path = os.path.join(artifacts_dir, 'feature_importance.png')
    fig.tight_layout()
    fig.savefig(fi_path)
    plt.close(fig)
    mlflow.log_artifact(fi_path, artifact_path="plots")

    # ROC-кривая
    fpr = metrics['fpr']
    tpr = metrics['tpr']
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", label="Случайная модель")
    ax.set_title("ROC-кривая")
    ax.set_xlabel("Доля ложных срабатываний")
    ax.set_ylabel("Доля правильных срабатываний")
    ax.legend()
    roc_path = os.path.join(artifacts_dir, 'roc_curve.png')
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)
    mlflow.log_artifact(roc_path, artifact_path="plots")

def register_best_model(**kwargs):
    """
    Регистрирует лучшую модель CatBoost в MLflow Model Registry под именем 'Catboost'.

    Аргументы:
        **kwargs: аргументы Airflow, содержит объект ti для XCom.
    """
    mlflow_run_id = kwargs["ti"].xcom_pull(task_ids="train_model")["run_id"]
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model("Catboost")
    client.create_model_version(
        name="Catboost",
        source=f"runs:/{mlflow_run_id}/model",
        run_id=mlflow_run_id,
    )
    _logger.info("Версия модели зарегистрирована!")
    
get_train_data_operator = PythonOperator(
    task_id="get_train_data",
    python_callable=get_train_data,
    dag=dag
)

check_data_quality_operator = BranchPythonOperator(
    task_id="check_data_quality",
    python_callable=check_data_quality,
    dag=dag
)

notify_bad_data_operator = EmailOperator(
    task_id="notify_bad_data",
    to=args['email'],
    subject="[ALERT] Bad training data",
    html_content="Check train data quality!",
)

train_model_operator = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

log_metrics_operator = PythonOperator(
    task_id="log_metrics",
    python_callable=log_metrics,
    dag=dag
)

log_artifacts_operator = PythonOperator(
    task_id="log_artifacts",
    python_callable=log_artifacts,
    dag=dag
)

notify_success_operator = EmailOperator(
    task_id="notify_success",
    to=args['email'],
    subject="[SUCCESS] Model successfuly trained",
    html_content="УРА УРА УРА!",
)

register_best_model_operator = PythonOperator(
    task_id="register_best_model",
    python_callable=register_best_model,
    trigger_rule=TriggerRule.ALL_SUCCESS,
)

wait_for_data_preparation = ExternalTaskSensor(
    task_id="wait_for_data_preparation_24h",
    external_dag_id="data_preparation",
    external_task_id="validate_loaded_data",
    execution_delta=timedelta(days=1),
    mode="reschedule",
    poke_interval=60,
    timeout=5*60,
)

get_train_data_operator >> check_data_quality_operator
check_data_quality_operator >> train_model_operator
check_data_quality_operator >> notify_bad_data_operator
train_model_operator >> [log_metrics_operator, log_artifacts_operator] >> register_best_model_operator
