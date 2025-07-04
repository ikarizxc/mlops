import argparse
import os
import pandas as pd
import logging

from sqlalchemy import create_engine

_logger = logging.getLogger(__name__)

def load_to_train_table(input_path: str) -> None:
    """
    Load task: считывает преобразованные данные и записывает их в таблицу 'train_data' в Postgres.
    
    Args:
        path (str):
            Путь к файлу с данными для загрузки.
    """    
    _logger.info(f"Path to transformed data file: {input_path}")

    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    df = pd.read_parquet(input_path)

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
    
parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='/temp/some_file')
args = parser.parse_args()

load_to_train_table(args.input)