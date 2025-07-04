from abc import ABC
import logging
from typing import List
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """
    Базовый класс для препроцессинга данных.
    Содержит общие шаги: удаление дубликатов, заполнение пропусков,
    кодирование категорий, обрезку выбросов и т.д.
    """

    def delete_high_correlation_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.85,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Удаляет признаки с корреляцией выше threshold.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            threshold (float): Порог корреляции.
            in_place (bool): Если True, изменяет df на месте, иначе возвращает копию.

        Returns:
            pd.DataFrame: DataFrame без высококоррелированных признаков.
        """
        target = df if in_place else df.copy()
        num_cols = target.select_dtypes(include=[np.number]).columns
        corr_matrix = target[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = upper.mean()[upper.mean() > threshold].index.tolist()

        _logger.info(f"Dropped high-corr features: {to_drop}")
        target.drop(columns=to_drop, inplace=True)
        return target

    def dummy_encode_categorical_features(
        self,
        df: pd.DataFrame,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Кодирует категориальные признаки в дамми-переменные.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            in_place (bool): Если True, изменяет df на месте, иначе возвращает копию.

        Returns:
            pd.DataFrame: DataFrame с дамми-признаками.
        """
        target = df if in_place else df.copy()
        cat_cols = target.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        _logger.debug(f"Encoding categorical columns: {cat_cols}")
        encoded = pd.get_dummies(
            target,
            columns=cat_cols,
            drop_first=True,
            dummy_na=False,
            dtype=int
        )
        _logger.info(f"Dummy encoding completed for columns: {cat_cols}")
        return encoded

    def cap_outliers(
        self,
        df: pd.DataFrame,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Ограничивает выбросы числовых признаков в пределах 1.5 * IQR.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            in_place (bool): Если True, изменяет df на месте, иначе возвращает копию.

        Returns:
            pd.DataFrame: DataFrame с ограниченными выбросами.
        """
        target = df if in_place else df.copy()
        num_cols = target.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            ser = target[col].astype(float)
            q1, q3 = ser.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            before = ((ser < low) | (ser > high)).sum()
            target[col] = ser.clip(lower=low, upper=high)
            _logger.debug(f"Capped {before} outliers in '{col}' to [{low}, {high}]")
        _logger.info(f"Outlier capping completed for columns: {num_cols}")
        return target

    def delete_duplicates(
        self,
        df: pd.DataFrame,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Удаляет дублирующиеся строки.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            in_place (bool): Если True, изменяет df на месте, иначе возвращает копию.

        Returns:
            pd.DataFrame: DataFrame без дубликатов.
        """
        target = df if in_place else df.copy()
        before = len(target)
        target.drop_duplicates(inplace=True)
        removed = before - len(target)
        _logger.info(f"Removed {removed} duplicate rows")
        return target

    def fill_null_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        value,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Заполняет пропуски указанным значением.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            columns (List[str]): Список столбцов.
            value: Значение для замены NaN.
            in_place (bool): Если True, изменяет df на месте, иначе возвращает копию.

        Returns:
            pd.DataFrame: DataFrame с заполненными пропусками.
        """
        target = df if in_place else df.copy()
        target[columns] = target[columns].fillna(value)
        _logger.info(f"Filled nulls for columns: {columns} with value: {value}")
        return target

    def get_categorical_features(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Возвращает список категориальных признаков.

        Args:
            df (pd.DataFrame): Входной DataFrame.

        Returns:
            List[str]: Имена категориальных столбцов.
        """
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        _logger.debug(f"Categorical features: {cat_cols}")
        return cat_cols
