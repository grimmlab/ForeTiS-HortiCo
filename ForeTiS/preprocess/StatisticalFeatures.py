import pandas as pd


def add_lagged_statistics(seasonal_periods: int, windowsize_lagged_statistics: int, seasonal_lags: int,
                          df: pd.DataFrame, features_sales: list, correlations: list):
    """
    Function adding lagged and seasonal-lagged features to dataset

    :param seasonal_periods: seasonal_period used for seasonal-lagged features
    :param windowsize_lagged_statistics: size of window used for sales statistics
    :param seasonal_lags: seasonal lags to add of the features specified
    :param df: dataset for adding features
    :param features_sales: sales features
    :param correlations: calculated correlations
    """
    if seasonal_lags == 0:
        print('No seasonal lags defined!')
    else:
        for seasonal_lag in range(seasonal_lags):
            seasonal_lag += 1
            if correlations is not None:
                df['stat_correlations_seaslag' + str(seasonal_lag) + '_sum'] = \
                    df[correlations].shift(seasonal_lag * seasonal_periods).sum(axis=1)
            for feature in features_sales:
                # separate function as different window sizes might be interesting compared to non-seasonal
                # statistics shift by 1+seasonal_period so rolling stats value is calculated without current value
                df['stat_' + feature + '_seaslag' + str(seasonal_lag)] = \
                    df[feature].shift(seasonal_lag * seasonal_periods)
                df['stat_' + feature + '_seaslag' + str(seasonal_lag) + '_rolling_mean' +
                        str(windowsize_lagged_statistics)] = df[feature].shift(seasonal_lag * seasonal_periods - 1).\
                    rolling(windowsize_lagged_statistics).mean().round(13)

def add_current_statistics(seasonal_periods: int, windowsize_current_statistics: int, df: pd.DataFrame,
                           features_weather_sales: list, correlations: list):
    """
    Function adding rolling seasonal statistics

    :param seasonal_periods: seasonal_period used for seasonal rolling statistics
    :param windowsize_current_statistics: size of window used for feature statistics
    :param df: dataset for adding features
    :param features_weather_sales: regex of the features of the dataset
    :param correlations: calculated correlations
    """
    if seasonal_periods <= windowsize_current_statistics:
        return
    if correlations is not None:
        df['stat_correlations_lag' + str(1) + '_sum'] = df[correlations].sum(axis=1)
    # separate function as different window sizes might be interesting compared to non-seasonal statistics
    for feature in features_weather_sales:
        df['stat_' + feature + '_lag' + str(1)] = df[feature].shift(1)
        df['stat_' + feature + '_rolling_mean' + str(windowsize_current_statistics)] = \
            df[feature].shift(1).rolling(windowsize_current_statistics).mean().round(13)


def add_current_weekday_statistics(windowsize_current_statistics: int, df: pd.DataFrame, features_sales: list):
    """
    Function adding rolling statistics for each week

    :param windowsize_current_statistics: size of window used for feature statistics
    :param df: dataset for adding features
    :param features_sales: n target column
    """
    weekday_indices = list()
    for day in range(0, 7):
        weekday_indices.append([index for index in df.index.date if index.weekday() == day])
    for indices in weekday_indices:
        for feature in features_sales:
            # shift by 1 so rolling statistics value is calculated without current value
            df.at[indices, 'stat_' + feature + '_weekday_rolling_mean' + str(windowsize_current_statistics)] = \
                df.loc[indices, feature].shift(1).rolling(windowsize_current_statistics).mean()
