import numpy as np
import pandas as pd
from scipy.stats import iqr, zscore

def missing_data(input_data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.

    input: pandas df
    output: pandas df

    '''

    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in input_data.columns:
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return (pd.DataFrame(table))


def iqr_outlier_threshold(df: pd.DataFrame, column: str) -> float:
    ''' Calculates the iqr outlier upper and lower thresholds

    input: pandas df, column of the same pandas df
    output: upper threshold, lower threshold

    '''
    iqr_value = iqr(df[column])
    lower_threshold = np.quantile(df[column], 0.25) - ((1.5) * (iqr_value))
    upper_threshold = np.quantile(df[column], 0.75) + ((1.5) * (iqr_value))
    print('Outlier threshold calculations:',
          f'IQR: {iqr_value}', f'Lower threshold:{lower_threshold}', f'Upper threshold: {upper_threshold}')

    return upper_threshold, lower_threshold


def mean_std_outliers(df: pd.DataFrame) -> float:
    """
    This function calculates the outlier threshold using the mean and standard deviation method

    input: pandas df

    output: upper threshold, lower threshold
    """

    z_score = zscore(df)
    upper_threshold = (z_score > 3).all(axis=1)
    lower_threshold = (z_score < -2).all(axis=1)
    upper = df[upper_threshold]
    lower = df[lower_threshold]
    print('Outlier threshold calculations:',
          f'Lower threshold:{lower}', f'Upper threshold: {upper}')

    return upper, lower
