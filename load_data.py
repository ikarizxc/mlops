import os
from typing import List
import pandas as pd
from sqlalchemy import Engine, create_engine
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger()

load_dotenv()


folder = 'data'
files_to_exclude = ['HomeCredit_columns_description.csv', 'sample_submission.csv']

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_ADDRESS = os.getenv('POSTGRES_ADDRESS')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')

engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_ADDRESS}:{POSTGRES_PORT}/{POSTGRES_DB}')

def get_csv_files(folder: str) -> List[str]:
    """
    Возвращает полные пути к файлам .csv из папки folder.

    Args:
        folder (str): путь к папке с файлами
    Returns:
        List[str]: список полных путей к .csv файлам
    """
    files_in_folder = os.listdir(folder)
    csv_paths = [
        os.path.join(folder, f)
        for f in files_in_folder
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.csv')
    ]
    return csv_paths

def load_csv_in_database(engine: Engine, csv_path: str) -> None:
    """
    Возвращает полные пути к файлам .csv из папки folder.

    Args:
        engine (sqlalchemy.engine.Engine): 
            SQLAlchemy Engine для подключения к базе данных.
        csv_path (str):
            Путь к .csv файлу.
    """
    table_name = csv_path.split('/')[-1].split('.')[0]
    _logger.info(f"Starting to load '{table_name}' table.")
    
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    _logger.info(f"Loaded '{table_name}' table.")
    
csv_files = get_csv_files(folder)

_logger.info(f"Found {len(csv_files)}: {[os.path.basename(f) for f in csv_files]}.")
_logger.info(f"Files to exlude: {files_to_exclude}.")

for f in csv_files:
    if os.path.basename(f) not in files_to_exclude:
        load_csv_in_database(engine, f)