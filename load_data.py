import argparse
import os
from typing import List
import pandas as pd
from dotenv import load_dotenv
import logging
from sqlalchemy import Engine, create_engine
from sqlalchemy_utils import database_exists, create_database

load_dotenv()
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger()

parser = argparse.ArgumentParser(
                    prog='Pg data loader',
                    description='Load .csv file to PostgreSQL')

parser.add_argument('--user', type=str, default='postgres')
parser.add_argument('--password', type=str, default='postgres')
parser.add_argument('--database', type=str, default='some_database')
parser.add_argument('--address', type=str, default='localhost')
parser.add_argument('--port', type=int, default=5432)

args = parser.parse_args()

folder = 'data'
files_to_exclude = ['HomeCredit_columns_description.csv', 'sample_submission.csv']

assert os.path.isdir(folder), f"Папка {folder} не найдена"

def get_engine(db_url: str) -> Engine:
    """Получить SQLAlchemy Engine для указанного URL базы данных, создав БД при необходимости.

    Эта функция проверяет существование базы по переданному URL. Если база не найдена,
    она будет создана (через `sqlalchemy_utils.create_database`). После этого
    возвращается готовый Engine для работы с БД.

    Args:
        db_url (str): URL подключения в формате
            `dialect+driver://username:password@host:port/database_name`, 
            например: `"postgresql://user:pass@localhost:5432/mydb"`.

    Returns:
        Engine: экземпляр `sqlalchemy.engine.Engine`, готовый для выполнения запросов.

    Raises:
        ValueError: если `db_url` пустой.
        sqlalchemy.exc.ArgumentError: при неверном формате URL.
        sqlalchemy.exc.OperationalError: при ошибках соединения.
    """
    if not db_url:
        raise ValueError("db_url must be a non-empty string")

    if not database_exists(db_url):
        create_database(db_url)
        print(f"Created database.")
    else:
        print(f"Database already exists.")

    return create_engine(db_url)

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

connection_string = f'postgresql://{args.user}:{args.password}@{args.address}:{args.port}/{args.database}'

engine = get_engine(connection_string)

for f in csv_files:
    if os.path.basename(f) not in files_to_exclude:
        load_csv_in_database(engine, f)