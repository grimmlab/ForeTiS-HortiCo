import pandas as pd

from ForeTiS.preprocess.DateCalenderFeatures import add_date_based_features, \
    add_valentine_mothersday, add_public_holiday_counters,add_school_holiday_counters
from ForeTiS.preprocess.StatisticalFeatures import add_current_statistics, \
    add_lagged_statistics, add_current_weekday_statistics


def add_calendar_features(df: pd.DataFrame, holiday_public_column: str, holiday_school_column: str, special_days: list,
                          cyclic_encoding: bool, resample_weekly: bool):
    """
    Function adding all calendar-based features

    :param df: dataset used for adding features
    :param holiday_public_column: name of the column containing the public holidays
    :param holiday_school_column: name of the column containing the school holidays
    :param special_days: special days for the specific data
    :param cyclic_encoding: whether cyclic encoding is done or not
    :param resample_weekly: whether to resample weekly or not
    """
    add_date_based_features(df=df, holiday_public_column=holiday_public_column, cyclic_encoding=cyclic_encoding)
    if special_days:
        add_valentine_mothersday(df=df, holiday_public_column=holiday_public_column, special_days=special_days)
    add_public_holiday_counters(df=df, holiday_public_column=holiday_public_column, special_days=special_days,
                                resample_weekly=resample_weekly)
    add_school_holiday_counters(df=df, holiday_school_column=holiday_school_column, resample_weekly=resample_weekly)


def add_statistical_features(seasonal_periods: int, windowsize_current_statistics: int,
                             windowsize_lagged_statistics: int, seasonal_lags: int, df: pd.DataFrame,
                             resample_weekly: bool, features_weather_sales: list, features_sales: list,
                             correlations: list):
    """Function adding all statistical features

    :param seasonal_periods: seasonality used for seasonal-based features
    :param windowsize_current_statistics: size of window used for feature statistics
    :param windowsize_lagged_statistics: size of window used for sales statistics
    :param seasonal_lags: seasonal lags to add of the features specified
    :param df: dataset used for adding features
    :param resample_weekly: whether to resample weekly or not
    :param features_weather_sales: features for statistics
    :param features_sales: sales features
    :param correlations: calculated correlations
    """
    add_lagged_statistics(seasonal_periods=seasonal_periods, windowsize_lagged_statistics=windowsize_lagged_statistics,
                          seasonal_lags=seasonal_lags, df=df, features_sales=features_sales, correlations=correlations)
    add_current_statistics(seasonal_periods=seasonal_periods,
                           windowsize_current_statistics=windowsize_current_statistics, df=df,
                           features_weather_sales=features_weather_sales, correlations=correlations)
    if not resample_weekly:
        add_current_weekday_statistics(windowsize_current_statistics=windowsize_current_statistics, df=df,
                                       features_sales=features_sales)
