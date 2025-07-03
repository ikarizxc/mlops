import logging
import numpy as np
import pandas as pd

from src.base_preprocessor import BasePreprocessor

_logger = logging.getLogger(__name__)

class ApplicationPreprocessor(BasePreprocessor):
    """
    Класс для препроцессинга данных application_[train|test].
    """
    def add_working_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет признак: заявка была совершена в рабочие часы (8–18) или нет.
        
        Args:
            df (pandas.DataFrame): DataFrame 
        """
        df['IS_HOURS_WORKING'] = (
            df['HOUR_APPR_PROCESS_START']
                .between(8, 18)
                .astype(int)
        )
        _logger.info("Added working hours.")
        return df
    
    def add_family_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет бинарный признак SINGLE_FAMILY_STATUS (вдова или не замужем - 1).
        
        Args:
            df (pandas.DataFrame): DataFrame 
        """
        df['SINGLE_FAMILY_STATUS'] = (
            df['NAME_FAMILY_STATUS']
            .isin(['Widow', 'Single / not married'])
            .astype('int8')
        )
        _logger.info("Added family status.")
        return df
       
    def add_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет признаки: соотношений кредитных величин.
        
        Args:
            df (pandas.DataFrame): DataFrame 
        """
        new_features = {
            'CREDIT_INCOME_RATIO': df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
            'ANNUITY_CREDIT_RATIO': df['AMT_ANNUITY'] / df['AMT_CREDIT'],
            'CREDIT_MONTHS': df['AMT_CREDIT'] / df['AMT_ANNUITY'],
            'INITIAL_CREDIT_PAY': df['AMT_GOODS_PRICE'] - df['AMT_CREDIT'],
        }
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        _logger.info("Added creadit features.")
        return df
        
    def add_agg_ext_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет признаки: агрегация EXT_SOURCE_{1,2,3}: min/max/mean/std/ratio/weighted.
        
        Args:
            df (pandas.DataFrame): DataFrame 
        """        
        new_features = {
            "EXT_SOURCE_MIN": df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1),
            "EXT_SOURCE_MAX": df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1),
            "EXT_SOURCE_MEAN": df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1),
            "EXT_SOURCE_STD": df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1),
            "EXT_SOURCE_MIN_MAX_DIV": 
                df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
                / df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1),
            "EXT_SOURCE_WEIGHTED": 
                (
                    df['EXT_SOURCE_1'] + 
                    5 * df['EXT_SOURCE_2'] + 
                    3 * df['EXT_SOURCE_3']
                 ) / 3
        }
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        _logger.info("Added aggregated extrernal sources.")
        return df
        
    def add_days_percents_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет признаки соотношения дней: Employment/Birth, Registration/Birth, Publish/Birth.
        
        Args:
            df (pandas.DataFrame): DataFrame 
        """
        new_features = {
            'DAYS_EMP_BIRTH_PERCENT': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
            'DAYS_REG_BIRTH_PERCENT': df['DAYS_REGISTRATION'] / df['DAYS_BIRTH'],
            'DAYS_PUB_BIRTH_PERCENT': df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH'],
        }
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        _logger.info("Added days precent features.")
        return df