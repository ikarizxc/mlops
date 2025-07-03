import os
import logging
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pendulum import today

from src.application_preprocessor import ApplicationPreprocessor

_logger = logging.getLogger(__name__)

args = {
    "owner": "Sviridyuk Sergey",
    "email": ["sergey.sviridyuk.41@gmail.com"],
    'email_on_failure': False,
}

dag = DAG(
    dag_id="data_preparation",
    default_args=args,
    max_active_runs=1,
    max_active_tasks=3,
    schedule="@daily",
    start_date=today("UTC").subtract(days=1),
    tags=["data", "preprocessing"],
)

def extract_raw_data(**kwargs) -> str:
    """
    Extract task: загружает исходные данные из PostgreSQL и сохраняет во временный Parquet файл.

    Returns:
        str: путь к созданному файлу Parquet с сырыми данными.
    """
    postgres_hook = PostgresHook(postgres_conn_id='home-credit-default-risk')
    sql = "SELECT * FROM application_train;"
    df: pd.DataFrame = postgres_hook.get_pandas_df(sql=sql)
    _logger.info(f"Loaded {len(df)} rows from 'application_train'")
    
    # мб в s3 загружать
    temp_directory = 'opt/airflow/tmp'
    os.makedirs(temp_directory, exist_ok=True)
    
    temp_path = f"{temp_directory}/raw_data.parquet"
    df.to_parquet(temp_path, index=False)
    _logger.info(f"Raw data temporary saved to '{temp_path}'")
    return temp_path

def transform_data(**kwargs) -> str:
    """
    Transform task: считывает Parquet файл, выполняет предобработку и сохраняет результат.

    Returns:
        str: путь к созданному файлу Parquet с преобразованными данными.
    """
    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids="task_extract_raw_data")
    
    df = pd.read_parquet(path)
    
    df_to_preprocess = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    df_not_to_preprocess = df[['SK_ID_CURR', 'TARGET']]
    
    applicaiton_preprocessor = ApplicationPreprocessor()
    applicaiton_preprocessor.cap_outliers(df_to_preprocess, in_place=True)
    applicaiton_preprocessor.delete_high_correlation_features(df_to_preprocess, in_place=True)
    applicaiton_preprocessor.fill_null_values(
        df_to_preprocess,
        applicaiton_preprocessor.get_categorical_features(df_to_preprocess),
        'UNKNOWN',
        in_place=True
    )
    df_to_preprocess = applicaiton_preprocessor.add_agg_ext_sources(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_credit_features(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_days_percents_features(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_family_status(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_working_hours(df_to_preprocess)
    
    df = pd.concat([df_not_to_preprocess, df_to_preprocess], axis=1)
    
    _logger.info("Data successfuly transformed")
    
    temp_path = "opt/airflow/tmp/transformed_data.parquet"
    df.to_parquet(temp_path, index=False)
    _logger.info(f"Transformed data temporary saved to '{temp_path}'")

    return temp_path
    
def load_to_train_table(**kwargs) -> None:
    """
    Load task: считывает преобразованные данные и записывает их в таблицу 'train_data' в Postgres.
    """    
    ti = kwargs['ti']
    path = ti.xcom_pull(task_ids="task_transform_data")
    _logger.info(path)
    df = pd.read_parquet(path)
    
    postgres_hook = PostgresHook(postgres_conn_id='home-credit-default-risk')
    engine = postgres_hook.get_sqlalchemy_engine()
    table_name = 'train_data'
    
    _logger.info(f"Starting to write {len(df)} rows to '{table_name}'")

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )
    _logger.info(f"Wrote {len(df)} rows to 'target_table'")

extract_raw_data_operator = PythonOperator(
    task_id='task_extract_raw_data',
    python_callable=extract_raw_data,
    dag=dag
)
transform_data_operator = PythonOperator(
    task_id='task_transform_data',
    python_callable=transform_data,
    dag=dag
)

load_to_train_table_operator = PythonOperator(
    task_id='task_load_to_train_table',
    python_callable=load_to_train_table,
    dag=dag
)

extract_raw_data_operator >> transform_data_operator >> load_to_train_table_operator