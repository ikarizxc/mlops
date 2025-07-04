import os
import pandas as pd
import logging
import argparse
from sqlalchemy import create_engine

_logger = logging.getLogger(__name__)

def extract_raw_data(output_path: str) -> None:
    """
    Extract task: загружает исходные данные из PostgreSQL и сохраняет во временный Parquet файл.
    
    Args:
        output_path (str):
            Путь где сохранится выгруженные данные из бд.
    """
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)
    df = pd.read_sql("SELECT * FROM application_train;", engine)
    _logger.info(f"Loaded {len(df)} rows from 'application_train'")
    
    df.to_parquet(output_path, index=False)
    _logger.info(f"Raw data temporary saved to '{output_path}'")

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str, default='/temp/some_file')
args = parser.parse_args()

extract_raw_data(args.output)