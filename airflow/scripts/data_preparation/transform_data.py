import pandas as pd
import logging
import argparse

from scripts.preprocessors.application_preprocessor import ApplicationPreprocessor

_logger = logging.getLogger(__name__)

def transform_data(input_path: str, output_path: str) -> None:
    """
    Transform task: считывает Parquet файл, выполняет предобработку и сохраняет результат.

    Args:
        input_path (str):
            Файл с исходными данными.
        output_path (str):
            Файл с преобразованными данными.
    """
    df = pd.read_parquet(input_path)
    
    df_to_preprocess = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    df_not_to_preprocess = df[['SK_ID_CURR', 'TARGET']]
    
    applicaiton_preprocessor = ApplicationPreprocessor()
    applicaiton_preprocessor.cap_outliers(df_to_preprocess, in_place=True)
    applicaiton_preprocessor.delete_high_correlation_features(df_to_preprocess, in_place=True)
    applicaiton_preprocessor.fill_null_values(
        df_to_preprocess,
        applicaiton_preprocessor.get_categorical_features(df_to_preprocess),
        'UNKNOWN',
        in_place=True
    )
    df_to_preprocess = applicaiton_preprocessor.add_agg_ext_sources(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_credit_features(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_days_percents_features(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_family_status(df_to_preprocess)
    df_to_preprocess = applicaiton_preprocessor.add_working_hours(df_to_preprocess)
    
    df = pd.concat([df_not_to_preprocess, df_to_preprocess], axis=1)
    
    _logger.info("Data successfuly transformed")

    df.to_parquet(output_path, index=False)
    _logger.info(f"Transformed data temporary saved to '{output_path}'")

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='/temp/some_file')
parser.add_argument('--output', type=str, default='/temp/some_file')
args = parser.parse_args()

transform_data(args.input, args.output)