import pandas as pd
import numpy as np
import os
import warnings
import configparser
import re
from sklearn.model_selection import train_test_split

from .raw_data_functions import custom_resampler, drop_columns, get_one_hot_encoded_df, impute_dataset_train_test
from . import FeatureAdder


class Dataset:
    """
    Class containing datasets ready for optimization.

    **Attributes**

        - target_column (*str*): the target column for the prediction
        - data_dir (*str*): data directory where the phenotype and genotype matrix are stored
        - data (*str*): the dataset that you want to use
        - windowsize_current_statistics (*int*): the windowsize for the feature engineering of the current statistic
        - windowsize_lagged_statistics (*int*): the windowsize for the feature engineering of the lagged statistics
        - seasonal_lags (*int*): the seasonal lags to add in the feature engineering for the lagged statistics
        - cyclic_encoding (*bool*): whether to do cyclic encoding or not
        - correlation_method (*str*): the used method to calculate the correlations
        - correlation_number (*int*): the number of with the focus product correlating products
        - datatype (*str*): if the data is in american or german type
        - date_column (*str*): the name of the column containg the date
        - group (*str*): if the data is from the old or API group
        - seasonal_periods (*int*): how many datapoints one season has
        - imputation (*bool*): whether to perfrom imputation or not
        - holiday_school_column (*str*): the column name containing the school holidays
        - holiday_public_column (*str*): the column name containing the public holidays
        - special_days (*list<str>*): the special days in your data
        - resample_weekly (*bool*): whether to resample weekly or not

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param data: the dataset that you want to use
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param target_column: the target column for the prediction
    :param windowsize_current_statistics: the windowsize for the feature engineering of the current statistic
    :param windowsize_lagged_statistics: the windowsize for the feature engineering of the lagged statistics
    :param seasonal_lags: the seasonal lags to add in the feature engineering for the lagged statistics
    :param cyclic_encoding: whether to do cyclic encoding or not
    :param imputation_method: the imputation method to use. Options are: 'mean' , 'knn' , 'iterative'
    :param correlation_number: the number of with the focus product correlating products
    :param correlation_method: the used method to calculate the correlations
    :param config: the information from dataset_specific_config.ini
    """

    def __init__(self, data_dir: str, data: str, test_set_size_percentage: int, target_column: str,
                 windowsize_current_statistics: int, windowsize_lagged_statistics: int, seasonal_lags: int,
                 cyclic_encoding: bool = False, imputation_method: str = 'None', correlation_number: int = None,
                 correlation_method: str = None, config: configparser.ConfigParser = None):
        self.target_column = target_column
        self.data_dir = data_dir
        self.data = data
        self.windowsize_current_statistics = windowsize_current_statistics
        self.windowsize_lagged_statistics = windowsize_lagged_statistics
        self.seasonal_lags = seasonal_lags
        self.cyclic_encoding = cyclic_encoding
        self.correlation_method = correlation_method
        self.correlation_number = correlation_number

        self.datatype = config[data]['datatype']
        self.date_column = config[data]['date_column']
        self.group = config[data]['group']
        self.seasonal_periods = config[data].getint('seasonal_periods')
        self.imputation = config[data].getboolean('imputation')
        self.holiday_school_column = config[data]['holiday_school_column']
        self.holiday_public_column = config[data]['holiday_public_column']
        self.special_days = config[data]['special_days'].replace(" ", "").replace("_", " ").split(',')
        self.resample_weekly = config[data].getboolean('resample_weekly')
        features_weather_regex = config[data]['features_weather_regex'].replace(" ", "").split(',')

        #  check if data is already preprocessed. If not, preprocess the data
        if os.path.exists(os.path.join(data_dir, data + '.h5')):
            datasets = list()
            with pd.HDFStore(os.path.join(data_dir, data + '.h5'), 'r') as hdf:
                keys = hdf.keys()
                for key in keys:
                    dataset = pd.read_hdf(os.path.join(data_dir, data + '.h5'), key=key)
                    dataset.name = key[1:]
                    datasets.append(dataset)
            if target_column in datasets[0]:
                print('---Dataset is already preprocessed---')
            else:
                raise Exception('Dataset was already preprocessed, but with another target column. '
                                'Please check target column again.')
        else:
            print('---Start preprocessing data---')

            # load raw data
            dataset_raw = self.load_raw_data(data_dir=data_dir, data=data)

            # sum up the turnovers if the data is from the API
            if self.group == 'API':
                if 'turnover' in self.target_column:
                    turnovers = []
                    for column in dataset_raw.columns:
                        if 'turnover' in column:
                            turnovers.append(column)
                    dataset_raw['total_turnover'] = dataset_raw[turnovers].sum(axis=1)
                    dataset_raw.drop(turnovers, axis=1, inplace=True)
                elif 'amount' in self.target_column:
                    self.correlations = \
                        self.get_corr(df=dataset_raw, test_set_size_percentage=test_set_size_percentage).index.tolist()

            dataset_raw = dataset_raw.asfreq('D')

            # condense columns if specified in config file
            if 'cols_to_condense' in config[data]:
                cols_to_condense = config[data]['cols_to_condense'].replace(" ", "").split(',')
                condensed_col_name = config[data]['condensed_col_name']
                dataset_raw[condensed_col_name] = 0
                for col in cols_to_condense:
                    dataset_raw[condensed_col_name] += dataset_raw[col]
                drop_columns(df=dataset_raw, columns=cols_to_condense)

            # create lists of the column names of the specific datasets
            searchfor = features_weather_regex
            features_weather = dataset_raw.filter(regex='|'.join(searchfor), axis=1).columns.tolist()
            self.features_holidays = [self.holiday_school_column] + [self.holiday_public_column]
            self.features = [self.target_column] + features_weather + self.features_holidays
            self.features_weather_sales = [self.target_column] + features_weather
            self.features_sales = [self.target_column]
            if hasattr(self, 'correlations'):
                self.features += self.correlations
                self.features_weather_sales += self.correlations
                self.features_sales += self.correlations

            # drop sales columns that are not target column and not useful columns
            dataset_raw = self.drop_non_target_useless_columns(df=dataset_raw)

            if self.imputation:
                dataset_raw = impute_dataset_train_test(df=dataset_raw,
                                                        test_set_size_percentage=test_set_size_percentage,
                                                        imputation_method=imputation_method)

            # set specific columns to datatype string
            self.set_dtypes(df=dataset_raw)

            # fill nans that are either no sale or no holiday
            if self.group == 'API':
                self.fill_nans_raw_data(df=dataset_raw)

            # add features, resample, and preprocess
            datasets = self.featureadding_and_resampling(df=dataset_raw)
            print('---Data preprocessed---')

        self.datasets = datasets

    def load_raw_data(self, data_dir: str, data: str) -> pd.DataFrame:
        """
        Load raw datasets

        :param data_dir: directory where the data is stored
        :param data: which dataset should be loaded

        :return: list of datasets to use for optimization
        """
        # load and name raw dataset
        if self.datatype == 'american':
            dataset_raw = pd.read_csv(os.path.join(data_dir, data + '.csv'), index_col=self.date_column)
        elif self.datatype == 'german':
            dataset_raw = pd.read_csv(os.path.join(data_dir, data + '.csv'), index_col=self.date_column,
                                      sep=';', decimal=',')
        dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%Y-%m-%d')

        return dataset_raw

    def drop_non_target_useless_columns(self, df: pd.DataFrame):
        """
        Drop the possible target columns that where not chosen as target column

        :param df: DataFrame to use for dropping

        :return: DataFrame with only the target column and features left
        """
        return df[self.features]

    def set_dtypes(self, df: pd.DataFrame):
        """
        Function setting dtypes of dataset. cols_to_str are converted to string, rest except date to float.
        Needed due to structure of raw file

        :param df: DataFrame whose columns data types should be set
        """
        for col in df.columns:
            if col in self.features_holidays:
                df[col] = df[col].astype(dtype='string')
            elif col != self.date_column:
                df[col] = df[col].astype(dtype='float')

    def fill_nans_raw_data(self, df: pd.DataFrame):
        """
        Fills the nans of days with no sells and no holiday

        :param df: DataFrame to fill

        :return: Filled DataFrame
        """
        for column in list(df.columns):
            if column in self.features_sales:
                df[column].fillna(0, inplace=True)
            elif column in self.features_holidays:
                df[column].fillna('no', inplace=True)

    def get_corr(self, df: pd.DataFrame, test_set_size_percentage: int) -> pd.Series:
        """
        Calculate the correlations

        :param df: DataFrame on which the correlations should be calculated
        :param test_set_size_percentage: the size of the test set in percentage

        :return: pd.Series with the top correlations
        """
        if test_set_size_percentage == 2021:
            test = df.loc['2021-01-01': '2021-12-31']
            train_val = pd.concat([df, test]).drop_duplicates(keep=False)
        else:
            train_val, _ = train_test_split(df, test_size=test_set_size_percentage * 0.01, random_state=42,
                                            shuffle=False)

        # filter only sold items
        data_amount = train_val.filter(regex="amount")
        data_amount_rs = data_amount.resample('W').sum()

        # Arranging and cutting data frames
        corr_p = data_amount_rs.corr(method=self.correlation_method)
        np.fill_diagonal(corr_p.values, 0)
        corr_p_filtered = corr_p.filter(regex=self.target_column)
        corr_p_filtered = corr_p_filtered.sort_values(by=[self.target_column], ascending=False)
        corr_p_top_n = corr_p_filtered.head(self.correlation_number)
        corr_p_top_n.reset_index().rename(columns={'index': 'corr_method'})
        if self.target_column in corr_p_top_n.index:
            corr_p_top_n.drop(self.target_column, axis=0, inplace=True)

        return corr_p_top_n

    def featureadding_and_resampling(self, df: pd.DataFrame) -> list:
        """
        Function preparing train and test sets for training based on raw dataset:
        - Feature Extraction
        (- Resampling if specified)
        - Deletion of non-target sales columns

        :param df: dataset with raw samples

        :return: Data with added features and resampling
        """
        print('--Adding calendar dataset--')
        FeatureAdder.add_calendar_features(df=df, holiday_public_column=self.holiday_public_column,
                                           holiday_school_column=self.holiday_school_column,
                                           special_days=self.special_days, cyclic_encoding=self.cyclic_encoding,
                                           resample_weekly=self.resample_weekly)
        print('--Added calendar dataset--')

        if not self.resample_weekly:
            print('-Adding statistical dataset-')
            # check if dataset is long enough for the given number of saesonal lags
            if max(self.seasonal_lags)*2 >= df.shape[0]/self.seasonal_periods:
                raise Exception('Dataset not long enough for the given number of seasonal lags. '
                                'Try less seasonal lags.')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  windowsize_current_statistics=self.windowsize_current_statistics,
                                                  windowsize_lagged_statistics=self.windowsize_lagged_statistics,
                                                  seasonal_lags=self.seasonal_lags, df=df,
                                                  resample_weekly=self.resample_weekly,
                                                  features_weather_sales=self.features_weather_sales,
                                                  features_sales=self.features_sales,
                                                  correlations=self.correlations+self.target_column
                                                  if hasattr(self, 'correlations') else None)
            if hasattr(self, 'correlations'):
                drop_columns(df=df, columns=self.correlations)
            print('-Added statistical dataset-')

        # one hot encode the data
        print('-one-hot-encoding the data-')
        df = get_one_hot_encoded_df(df=df, columns_to_encode=list(df.select_dtypes(include=['string']).columns))
        print('-one-hot-encoded the data-')

        # resample
        if self.resample_weekly:
            print('-Weekly resample data-')
            df = df.resample('W').apply(lambda x: custom_resampler(arraylike=x, target_column=self.target_column))
            df_2022 = df.loc['2022-01-01': '2022-12-31']
            df = pd.concat([df, df_2022]).drop_duplicates(keep=False)
            if 'cal_date_weekday' in df.columns:
                drop_columns(df=df, columns=['cal_date_weekday'])
            if 'cal_date_weekday_sin' in df.columns:
                drop_columns(df=df, columns=['cal_date_weekday_sin', 'cal_date_weekday_cos'])
            print('-Weekly resampled data-')

            # statistical feature extraction on dataset
            print('-Adding statistical dataset-')
            if self.seasonal_lags:
                if max(self.seasonal_lags)*2 >= df.shape[0]/self.seasonal_periods:
                    raise Exception('Dataset not long enough for the given number of seasonal lags. '
                                    'Try less seasonal lags.')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  windowsize_current_statistics=self.windowsize_current_statistics,
                                                  windowsize_lagged_statistics=self.windowsize_lagged_statistics,
                                                  seasonal_lags=self.seasonal_lags, df=df,
                                                  resample_weekly=self.resample_weekly,
                                                  features_weather_sales=self.features_weather_sales,
                                                  features_sales=self.features_sales,
                                                  correlations=self.correlations+[self.target_column]
                                                  if hasattr(self, 'correlations') else None)
            if hasattr(self, 'correlations'):
                drop_columns(df=df, columns=self.correlations)
            print('-Added statistical dataset-')

        # drop a column if it only contains nans
        for column in df:
            if df[column].isnull().all():
                warnings.warn("Warning: Will drop one or more statistical features due to only containing NaNs")
                df = df.dropna(axis=1, how='all')
                break

        # drop columns that stay constant
        drop_columns(df=df, columns=df.columns[df.nunique() <= 1])

        # drop missing values after adding statistical features (e.g. due to lagged features)
        df.dropna(inplace=True)

        if hasattr(self, 'correlations'):
            for column in df.columns:
                if re.search('stat_correlations.*', column):
                    self.correlations += [column]
        if self.group == 'API':
            dataset_weather = pd.concat([df.filter(regex='whi'), df[self.target_column]], axis=1)
        else:
            dataset_weather = df[df.columns.drop(list(df.filter(regex='cal|holiday|stat_[A-Z]')))]
        dataset_weather.name = 'dataset_weather'
        if self.group == 'API':
            dataset_cal = pd.concat([df[self.target_column], df.filter(regex='cal'),
                                     df.filter(regex='^holiday_school')], axis=1)
        else:
            dataset_cal = pd.concat([df[self.target_column], df.filter(regex='cal'),
                                     df.filter(regex='^school_holiday')], axis=1)
        dataset_cal.name = 'dataset_cal'
        if self.group == 'API':
            dataset_sales_corr = pd.concat([df[self.target_column], df.filter(regex='stat')], axis=1)\
                .drop(list(df.filter(regex='stat_whi')), axis=1)
            dataset_sales = dataset_sales_corr.copy()
            if hasattr(self, 'correlations'):
                for column in self.correlations:
                    dataset_sales.drop(list(dataset_sales.filter(regex=column)), axis=1, inplace=True)
        else:
            dataset_sales = pd.concat([df[self.target_column], df.filter(regex='stat')], axis=1) \
                .drop(list(df.filter(regex='stat_[a-z]')), axis=1)
        dataset_sales.name = 'dataset_sales'
        if hasattr(self, 'correlations'):
            dataset_sales_corr.name = 'dataset_sales_corr'
            dataset_weather_sales_corr = pd.concat(
                [dataset_weather, dataset_sales_corr.drop(self.target_column, axis=1)],
                axis=1)
            dataset_weather_sales_corr.name = 'dataset_weather_sales_corr'
            dataset_cal_sales_corr = pd.concat([dataset_cal, dataset_sales_corr.drop(self.target_column, axis=1)],
                                               axis=1)
            dataset_cal_sales_corr.name = 'dataset_cal_sales_corr'
        dataset_weather_sales = pd.concat([dataset_weather, dataset_sales.drop(self.target_column, axis=1)], axis=1)
        dataset_weather_sales.name = 'dataset_weather_sales'
        dataset_weather_cal = pd.concat([dataset_weather, dataset_cal.drop(self.target_column, axis=1)], axis=1)
        dataset_weather_cal.name = 'dataset_weather_cal'
        dataset_cal_sales = pd.concat([dataset_cal, dataset_sales.drop(self.target_column, axis=1)], axis=1)
        dataset_cal_sales.name = 'dataset_cal_sales'
        if hasattr(self, 'correlations'):
            dataset_full_corr = df.copy()
            dataset_full_corr.name = 'dataset_full_corr'
        dataset_full = df.copy()
        if hasattr(self, 'correlations'):
            for column in self.correlations:
                dataset_full.drop(list(dataset_full.filter(regex=column)), axis=1, inplace=True)
        dataset_full.name = 'dataset_full'

        filename_h5 = os.path.join(self.data_dir, self.data + '.h5')
        dataset_weather.to_hdf(filename_h5, key='dataset_weather')
        dataset_cal.to_hdf(filename_h5, key='dataset_cal')
        dataset_sales.to_hdf(filename_h5, key='dataset_sales')
        if hasattr(self, 'correlations'):
            dataset_sales_corr.to_hdf(filename_h5, key='dataset_sales_corr')
            dataset_weather_sales_corr.to_hdf(filename_h5, key='dataset_weather_sales_corr')
            dataset_cal_sales_corr.to_hdf(filename_h5, key='dataset_cal_sales_corr')
            dataset_full_corr.to_hdf(filename_h5, key='dataset_full_corr')
        dataset_weather_sales.to_hdf(filename_h5, key='dataset_weather_sales')
        dataset_weather_cal.to_hdf(filename_h5, key='dataset_weather_cal')
        dataset_cal_sales.to_hdf(filename_h5, key='dataset_cal_sales')
        dataset_full.to_hdf(filename_h5, key='dataset_full')

        if hasattr(self, 'correlations'):
            datasets = [dataset_weather, dataset_cal, dataset_sales, dataset_sales_corr, dataset_weather_sales,
                        dataset_weather_sales_corr, dataset_weather_cal, dataset_cal_sales, dataset_cal_sales_corr,
                        dataset_full, dataset_full_corr]
        else:
            datasets = [dataset_weather, dataset_cal, dataset_sales, dataset_weather_sales, dataset_weather_cal,
                        dataset_cal_sales, dataset_full]

        return datasets
