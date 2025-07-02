import argparse
import os
from typing import List
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(
    description="Data loader to postgres from csv files"
)
parser.add_argument(
    "-a", "--address",
    type=str,
    default='localhost',
)
parser.add_argument(
    "-p", "--port",
    type=int,
    default=5432,
)
args = parser.parse_args()

folder = 'data'
files_to_exclude = ['HomeCredit_columns_description', 'sample_submission']

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{args.address}:{args.port}/{POSTGRES_DB}')

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

def load_csv_in_database(engine, csv_path: str) -> None:
    table_name = csv_path.split('/')[-1].split('.')[0]
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    
csv_files = get_csv_files(folder)

for f in csv_files:
    if os.path.basename(f) not in files_to_exclude:
        load_csv_in_database(engine, f)