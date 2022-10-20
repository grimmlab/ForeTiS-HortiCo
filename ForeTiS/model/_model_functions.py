import joblib
import pandas as pd

from . import _base_model


def load_retrain_model(path: str, filename: str, retrain: pd.DataFrame, early_stopping_point: int = None) \
        -> _base_model.BaseModel:
    """
    Load and retrain persisted model
    :param path: path where the model is saved
    :param filename: filename of the model
    :param retrain: data for retraining
    :param early_stopping_point: optional early stopping point relevant for some models
    :return: model instance
    """
    model = load_model(path=path, filename=filename)
    if early_stopping_point is not None:
        model.early_stopping_point = early_stopping_point
    model.prediction = None
    model.retrain(retrain=retrain)
    return model


def load_model(path: str, filename: str) -> _base_model.BaseModel:
    """
    Load persisted model
    :param path: path where the model is saved
    :param filename: filename of the model
    :return: model instance
    """
    path = path + '/' if path[-1] != '/' else path
    model = joblib.load(path + filename)
    return model
