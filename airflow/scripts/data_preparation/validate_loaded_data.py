import os
import pandas as pd
import logging

from sqlalchemy import create_engine

_logger = logging.getLogger(__name__)

def validate_data_loaded() -> None:
    """
    Validation task: проверяет целостность данных после загрузки (непустая таблица).
    """
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)
    df = pd.read_sql("SELECT COUNT(*) AS cnt FROM train_data;", engine)
    cnt = int(df.loc[0, 'cnt'])
    if cnt < 1:
        raise ValueError("Validation error: 'train_data' table is empty.")
    _logger.info(f"Validation OK: {cnt} rows")

validate_data_loaded()