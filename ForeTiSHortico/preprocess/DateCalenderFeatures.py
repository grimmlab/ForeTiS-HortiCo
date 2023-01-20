import pandas as pd
import math

from ForeTiSHortico.preprocess.raw_data_functions import drop_columns, encode_cyclical_features


def add_date_based_features(df: pd.DataFrame, holiday_public_column: str, cyclic_encoding: bool):
    """
    Function adding date based features to dataset

    :param df: dataset for adding features
    :param holiday_public_column: name of the column containing the public holidays
    :param cyclic_encoding: whether cyclic encoding is done or not
    """
    df['cal_date_day_of_month'] = df.index.day
    df['cal_date_weekday'] = df.index.weekday
    df['cal_date_month'] = df.index.month
    for index in df.index:
        if index.weekday() == 6 or df.loc[index, holiday_public_column] != 'no':
            df.at[index, 'cal_date_workingday'] = False
        else:
            df.at[index, 'cal_date_workingday'] = True
    df['cal_date_workingday'] = df['cal_date_workingday'].astype(dtype='bool')
    if cyclic_encoding == True:
        encode_cyclical_features(df=df, columns=['cal_date_day_of_month', 'cal_date_weekday', 'cal_date_month'])


def add_valentine_mothersday(df: pd.DataFrame, holiday_public_column: str, special_days: list):
    """
    Function adding valentine's and mother's day to public_holiday column of dataset

    :param df: dataset for adding valentine's and mother's day
    :param holiday_public_column: name of the column containing the public holidays
    :param special_days: special days for the specific data
    """
    if 'Valentine' in special_days:
        # add valentine's day (always 14th of February)
        for index in df.index:
            if (index.day == 14 and index.month == 2):
                df.at[index, holiday_public_column] = 'Valentine'
    if 'MothersDay' in special_days:
        # add mother's day (in Germany always second sunday in May)
        for index in df.index:
            if ((index.day - 7) > 0 and index.day < 15 and index.weekday() == 6 and index.month == 5):
                df.at[index, holiday_public_column] = 'MothersDay'


def add_public_holiday_counters(df: pd.DataFrame, holiday_public_column: str, special_days: list,
                                resample_weekly: bool):
    """
    Function adding counters for upcoming or past public holidays (according to event_lags)
    with own counters for those specified in special_days

    :param df: dataset for adding features
    :param holiday_public_column: name of the column containing the public holidays
    :param special_days: special days for the specific data
    :param resample_weekly: whether to resample weekly or not
    """
    if resample_weekly:
        event_lags = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1,
                      2, 3, 4, 5, 6, 7]
        for index, row in df.iterrows():
            holiday = row[holiday_public_column]
            if holiday != 'no':
                for lag in event_lags:
                    if (index + pd.Timedelta(days=lag)) in df.index:
                        df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday_public_column + '_Counter'] = -math.ceil(lag/7)
                        if holiday in special_days:
                            df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -math.ceil(lag/7)
    else:
        event_lags = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
        for index, row in df.iterrows():
            holiday = row[holiday_public_column]
            if holiday != 'no':
                for lag in event_lags:
                    if (index+pd.Timedelta(days=lag)) in df.index:
                        df.at[index+pd.Timedelta(days=lag), 'cal_' + holiday_public_column + '_Counter'] = -lag
                        if holiday in special_days:
                            df.at[index+pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -lag
    drop_columns(df=df, columns=[holiday_public_column])
    df[[col for col in df.columns if 'Counter' in col]] = \
        df[[col for col in df.columns if 'Counter' in col]].fillna(value=99)


def add_school_holiday_counters(df: pd.DataFrame, holiday_school_column: str, resample_weekly: bool):
    """
    Function adding counters for upcoming or past public holidays (according to event_lags)
    with own counters for those specified in special_days

    :param df: dataset for adding features
    :param holiday_school_column: name of the column containing the public holidays
    :param resample_weekly: whether to resample weekly or not
    """
    current_holiday = None
    if resample_weekly:
        event_lags_past = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
        event_lags_future = [0, 1, 2, 3, 4, 5, 6, 7]
        for index, row in df.iterrows():
            holiday = row[holiday_school_column]
            if holiday != 'no':
                if holiday == current_holiday:
                    for lag in event_lags_future:
                        if (index + pd.Timedelta(days=lag)) in df.index:
                            df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -math.ceil(lag / 7)
                else:
                    current_holiday = holiday
                    for lag in event_lags_past:
                        if (index + pd.Timedelta(days=lag)) in df.index:
                            df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -math.ceil(lag/7)
    else:
        event_lags_past = [-7, -6, -5, -4, -3, -2, -1, 0]
        event_lags_future = [0, 1, 2, 3]
        for index, row in df.iterrows():
            holiday = row[holiday_school_column]
            if holiday != 'no':
                if holiday == current_holiday:
                    for lag in event_lags_future:
                        if (index + pd.Timedelta(days=lag)) in df.index:
                            df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -math.ceil(lag / 7)
                else:
                    current_holiday = holiday
                    for lag in event_lags_past:
                        if (index + pd.Timedelta(days=lag)) in df.index:
                            df.at[index + pd.Timedelta(days=lag), 'cal_' + holiday + '_Counter'] = -math.ceil(lag / 7)
    drop_columns(df=df, columns=[holiday_school_column])
    df[[col for col in df.columns if 'Counter' in col]] = \
        df[[col for col in df.columns if 'Counter' in col]].fillna(value=99)

