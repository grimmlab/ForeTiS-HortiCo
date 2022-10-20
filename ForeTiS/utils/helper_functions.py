import os
import inspect
import importlib
import pandas as pd
import torch
import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit
import ForeTiS.model


def get_list_of_featuresets() -> list:
    """
    Get a list of all possible featuresets.

    ! Adapt if new featuresets is added !

    :return: List of all possible featuresets
    """
    return ['optimize', 'dataset_weather', 'dataset_cal', 'dataset_sales', 'dataset_sales_corr',
            'dataset_weather_sales', 'dataset_weather_sales_corr', 'dataset_weather_cal', 'dataset_cal_sales',
            'dataset_cal_sales_corr', 'dataset_full', 'dataset_full_corr']


def get_list_of_implemented_models() -> list:
    """
    Create a list of all implemented models based on files existing in 'model' subdirectory of the repository.
    """
    # Assumption: naming of python source file is the same as the model name specified by the user
    if os.path.exists('../model'):
        model_src_files = os.listdir('../model')
    elif os.path.exists('model'):
        model_src_files = os.listdir('model')
    elif os.path.exists('ForeTiS/model'):
        model_src_files = os.listdir('ForeTiS/model')
    else:
        model_src_files = [model_file + '.py' for model_file in ForeTiS.model.__all__]
    model_src_files = [file for file in model_src_files if file[0] != '_']
    return [model[:-3] for model in model_src_files]


def get_mapping_name_to_class() -> dict:
    """
    Get a mapping from model name (naming in package model without .py) to class name.
    :return: dictionary with mapping model name to class name
    """
    if os.path.exists('../model'):
        files = os.listdir('../model')
    elif os.path.exists('model'):
        files = os.listdir('model')
    elif os.path.exists('ForeTiS/model'):
        files = os.listdir('ForeTiS/model')
    else:
        files = [model_file + '.py' for model_file in ForeTiS.model.__all__]
    modules_mapped = {}
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = 'ForeTiS.model.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    modules_mapped[file_name] = cls
    return modules_mapped

def set_all_seeds(seed: int=0):
    """
    Set all seeds of libs with a specific function for reproducibility of results
    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)

def get_folds(datasplit: str, n_splits: int):
    """
    Get the folds for the optuna optimization
    :param datasplit: which datasplit should be performed
    :param n_splits: number of splits for the timeseries-cv
    :return: number of folds
    """
    if datasplit == "timeseries-cv" or datasplit == "cv":
        folds = n_splits
    elif datasplit == "train-val-test":
        folds = 1

    return folds


def get_indexes(df: pd.DataFrame, n_splits: str, datasplit: str):
    """
    Get the indexes for timeseries-cv
    :param df: data that should be splited
    :param n_splits: number of splits for the timeseries-cv
    :param datasplit: splitting method
    :return: train and test indexes
    """
    if datasplit == 'timeseries-cv':
        splitter = TimeSeriesSplit(n_splits=n_splits)
    if datasplit == 'cv':
        splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    train_indexes = []
    test_indexes = []
    for train_index, test_index in splitter.split(df):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes



