import os
import logging
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pendulum import today

args = {
    "owner": "Sviridyuk Sergey",
    "email": ["sergey.sviridyuk.41@gmail.com"],
    'email_on_failure': False,
}

with DAG(
    dag_id="data_preparation_docker",
    default_args=args,
    max_active_runs=1,
    max_active_tasks=3,
    schedule="@daily",
    start_date=today("UTC").subtract(days=1),
    tags=["data", "preprocessing", "docker"],
) as dag:
    
    postgres_hook = PostgresHook(postgres_conn_id='home-credit-default-risk')
    connection_string = postgres_hook.sqlalchemy_url
    
    image_name = 'data-preparation-container:latest'
    
    host_tmp_dir = os.path.abspath('/tmp')
    container_tmp_dir = '/tmp'
    
    os.makedirs(host_tmp_dir, exist_ok=True)
    
    # --- CHECK CONNECTION TASK ---
    check_db_connection_operator = DockerOperator(
        task_id="task_check_db_connection",
        image=image_name,
        auto_remove='success',
        mount_tmp_dir=False,
        command=f'python3 scripts/data_preparation/check_db_connection.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        environment={
            "DATABASE_URL": connection_string
        },
    )    

    # --- EXTRACT TASK ---
    extract_raw_data_operator = DockerOperator(
        task_id="task_extract_raw_data",
        image=image_name,
        auto_remove='success',
        mount_tmp_dir=False,
        command=f'python3 scripts/data_preparation/extract_raw_data.py --output {container_tmp_dir}/raw.parquet',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {'source': host_tmp_dir,    'target': container_tmp_dir,    'type': 'bind'},
        ],
        environment={
            "DATABASE_URL": connection_string
        },
    )    
    
    # --- TRANSFORM TASK ---
    transform_data_operator = DockerOperator(
        task_id="task_transform_data",
        image=image_name,
        auto_remove='success',
        mount_tmp_dir=False,
        command=f'python3 scripts/data_preparation/transform_data.py --input {container_tmp_dir}/raw.parquet --output {container_tmp_dir}/transformed_data.parquet',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {'source': host_tmp_dir,    'target': container_tmp_dir,    'type': 'bind'},
        ],
    )   
        
    # --- LOAD TASK ---
    load_to_train_table_operator = DockerOperator(
        task_id="task_load_to_train_table",
        image=image_name,
        auto_remove='success',
        mount_tmp_dir=False,
        command=f'python3 scripts/data_preparation/load_to_train_table.py --input {container_tmp_dir}/transformed_data.parquet',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {'source': host_tmp_dir,    'target': container_tmp_dir,    'type': 'bind'},
        ],
        environment={
            "DATABASE_URL": connection_string
        },
    )
    
    # --- VALIDATE DATA TASK ---
    validate_loaded_data_operator = DockerOperator(
        task_id="task_validate_loaded_data",
        image=image_name,
        auto_remove='success',
        mount_tmp_dir=False,
        command=f'python3 scripts/data_preparation/validate_loaded_data.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        environment={
            "DATABASE_URL": connection_string
        },
    )  
        
    check_db_connection_operator \
        >> extract_raw_data_operator \
            >> transform_data_operator \
                >> load_to_train_table_operator \
                    >> validate_loaded_data_operator