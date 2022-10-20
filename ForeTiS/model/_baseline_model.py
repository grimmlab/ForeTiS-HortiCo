from . import _base_model
import abc
import pandas as pd
import numpy as np
import optuna
import sklearn


class BaselineModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all models with a sklearn-like API to share functionalities
    See BaseModel for more information
    """
    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, test_set_size_percentage: int,
                 featureset: str, target_column: str = None, current_model_name: str = None):
        self.target_column = target_column
        self.current_model_name = current_model_name
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset=featureset,
                         test_set_size_percentage=test_set_size_percentage, target_column=target_column)

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with sklearn-like API.
        See BaseModel for more information
        """
        observed_period = retrain.tail(self.window) if hasattr(self, 'window') else retrain
        self.average = observed_period[self.target_column].mean()

        if self.prediction is not None:
            if len(observed_period[self.target_column]) > len(self.prediction):
                y_true = observed_period[self.target_column][-len(self.prediction):]
                y_pred = self.prediction
            else:
                y_true = observed_period[self.target_column]
                y_pred = self.prediction[-len(observed_period[self.target_column]):]
        else:
            y_true = np.array([0])
            y_pred = np.array([0])
        self.var_artifical = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        :param update: data for updating
        :param period: the current refit cycle
        """
        observed_period = update.tail(self.window) if hasattr(self, 'window') else update
        self.average = observed_period[self.target_column].mean()

        y_true = observed_period[self.target_column][-len(self.prediction):]
        y_pred = self.prediction
        self.var_artifical = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See BaseModel for more information
        """
        # use average of train set for insample prediction (insample -> knowledge of whole train set)
        self.prediction = np.full((X_in.shape[0],), self.average)
        return self.prediction, self.var_artifical

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See BaseModel for more information
        """
        # train model
        self.prediction = None
        self.retrain(train)
        # validate model
        return self.predict(X_in=val)