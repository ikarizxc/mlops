import os
import logging

from sqlalchemy import create_engine, text

_logger = logging.getLogger(__name__)

def check_db_connection() -> None:
    """
    Connection check task: выполняет простейший запрос к Postgres для проверки доступности.
    """
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)
    # открываем соединение через SQLAlchemy Core
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1')).scalar()
    engine.dispose()

    if result == 1:
        _logger.info("Postgres connection OK.")
    else:
        raise ValueError("Postgres connection test failed, got: %s", result)

check_db_connection()