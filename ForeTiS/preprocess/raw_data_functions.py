import pandas as pd
import datetime
import sklearn.impute
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.experimental import enable_iterative_imputer


def drop_columns(df: pd.DataFrame, columns: list):
    """
    Function dropping all columns specified

    :param df: dataset used for dropping
    :param columns: columns which should be dropped
    """
    df.drop(columns=columns, inplace=True)


def drop_rows_by_dates(df: pd.DataFrame, start: datetime.date, end: datetime.date):
    """
    Function dropping rows within specified dates

    :param df: dataset used for dropping
    :param start: start date for dropped period
    :param end: end date for dropped period
    """
    df.drop(pd.date_range(start=start, end=end), inplace=True)


def custom_resampler(arraylike: pd.Series, target_column: str):
    """
    Custom resampling function when resampling frequency of dataset

    :param arraylike: Series to use for calculation
    :param target_column: choosen target column

    :return: sum or mean of arraylike or 1
    """
    if arraylike.name == target_column:
        return np.sum(arraylike)
    else:
        return np.mean(arraylike)


def get_one_hot_encoded_df(df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """
    Function delivering dataframe with specified columns one hot encoded

    :param df: dataset to use for encoding
    :param columns_to_encode: columns to encode

    :return: dataset with encoded columns
    """
    return pd.get_dummies(df, columns=columns_to_encode)


def impute_dataset_train_test(df: pd.DataFrame = None, test_set_size_percentage: float = 20,
                              imputation_method: str = None) -> pd.DataFrame:
    """
    Get imputed dataset as well as train and test set (fitted to train set)

    :param df: dataset to impute
    :param test_set_size_percentage: the size of the test set in percentage
    :param imputation_method: specify the used method if imputation is applied

    :return: imputed dataset, train and test set
    """
    cols_to_impute = df.loc[:, df.isna().any()].select_dtypes(exclude=['string', 'object']).columns.tolist()
    if len(cols_to_impute) == 0:
        return df
    cols_to_add = [col for col in df.columns.tolist() if col not in cols_to_impute]

    if test_set_size_percentage == 2021:
        test = df.loc['2021-01-01': '2021-12-31']
        train_val = pd.concat([df, test]).drop_duplicates(keep=False)
    else:
        train_val, _ = train_test_split(df, test_size=test_set_size_percentage * 0.01, random_state=42, shuffle=False)

    if imputation_method == 'mean':
        imputer = get_simple_imputer(df=train_val.filter(cols_to_impute))
    if imputation_method == 'knn':
        imputer = get_knn_imputer(df=train_val.filter(cols_to_impute))
    if imputation_method == 'iterative':
        imputer = get_iter_imputer(df=train_val.filter(cols_to_impute))
    data = imputer.transform(X=df.filter(cols_to_impute))
    dataset_imp = pd.concat([pd.DataFrame(data=data,
                                          columns=cols_to_impute, index=df.index), df[cols_to_add]],
                            axis=1, sort=False)
    return dataset_imp


def get_simple_imputer(df: pd.DataFrame, strategy: str = 'mean') -> sklearn.impute.SimpleImputer:
    """
    Get simple imputer for each column according to specified strategy

    :param df: DataFrame to impute
    :param strategy: strategy to use, e.g. 'mean' or 'median'

    :return: imputer
    """
    simple_imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=strategy)
    simple_imputer.fit(X=df)
    return simple_imputer


def get_iter_imputer(df: pd.DataFrame, sample_posterior: bool = True, max_iter: int = 100,
                     min_value: int = 0, max_value: int = None) -> sklearn.impute.IterativeImputer:
    """
    Multivariate, iterative imputer fitted to df with specified parameters

    :param df: DataFrame to fit for imputation
    :param sample_posterior: sample from predictive posterior of fitted estimator (standard: BayesianRidge())
    :param max_iter: maximum number of iterations for imputation
    :param min_value: min value for imputation
    :param max_value: max value for imputation

    :return: imputer
    """
    iterative_imputer = sklearn.impute.IterativeImputer(sample_posterior=sample_posterior, max_iter=max_iter,
                                                        min_value=min_value, max_value=max_value,
                                                        random_state=0)
    iterative_imputer.fit(X=df)
    return iterative_imputer


def get_knn_imputer(df: pd.DataFrame, n_neighbors: int = 10) -> sklearn.impute.KNNImputer:
    """
    Imputer of missing values according to k-nearest neighbors in feature space

    :param df: DataFrame to use for imputation
    :param n_neighbors: number of neighbors to use for imputation

    :return: imputer
    """
    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=n_neighbors)
    knn_imputer.fit(X=df)
    return knn_imputer


def encode_cyclical_features(df: pd.DataFrame, columns: list):
    """
    Function that encodes the cyclic features to sinus and cosinus distribution

    :param df: DataFrame to use for imputation
    :param columns: columns that should be encoded
    """
    for col in columns:
        max_val = df[col].max()
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        drop_columns(df=df, columns=col)
