from . import _base_model
import abc
import pandas as pd
import numpy as np
import optuna
import sklearn
import tensorflow as tf


class TensorflowModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~ForeTiS.model._base_model.BaseModel` for all TensorFlow models to share functionalities.
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.

    **Attributes**

        *Inherited attributes*

        See :obj:`~easypheno.model._base_model.BaseModel`.

        *Additional attributes*

        - x_scaler (*sklearn.preprocessing.StandardScaler*): Standard scaler for the x data
        - y_scaler (*sklearn.preprocessing.StandardScaler*): Standard scaler for the y data

    :param optuna_trial: Trial of optuna for optimization
    :param datasets: all datasets that are available
    :param featureset: on which featuresets the models should be optimized
    :param test_set_size_percentage: the size of the test set in percentage
    :param current_model_name: name of the current model according to naming of .py file in package model
    :param target_column: the target column for the prediction
    """
    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset: str, test_set_size_percentage: int,
                 target_column: str = None):
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset=featureset,
                         test_set_size_percentage=test_set_size_percentage, target_column=target_column)

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        x_train = retrain.drop(self.target_column, axis=1).values.reshape(-1, retrain.shape[1] - 1)
        y_train = retrain[self.target_column].values.reshape(-1, 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.fit_transform(y_train)

        self.model.data = (tf.convert_to_tensor(value=x_train.astype(float), dtype=tf.float64),
                           tf.convert_to_tensor(value=y_train.astype(float), dtype=tf.float64))
        self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)

        if self.prediction is not None:
            if len(retrain[self.target_column]) > len(self.prediction):
                y_true = retrain[self.target_column][-len(self.prediction):]
                y_pred = self.prediction
            else:
                y_true = retrain[self.target_column]
                y_pred = self.prediction[-len(retrain[self.target_column]):]
        else:
            y_true = np.array([0])
            y_pred = np.array([0])
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        x_train = update.drop(self.target_column, axis=1).values.reshape(-1, update.shape[1] - 1)
        y_train = update[self.target_column].values.reshape(-1, 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            self.y_scaler = sklearn.preprocessing.StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train)

        self.model.data = (tf.convert_to_tensor(value=x_train.astype(float), dtype=tf.float64),
                           tf.convert_to_tensor(value=y_train.astype(float), dtype=tf.float64))
        self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)

        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.inverse_transform(y_train)

        y_true = y_train[-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        X_in = X_in.drop(self.target_column, axis=1).values.reshape(-1, X_in.shape[1] - 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            X_in = self.x_scaler.transform(X_in)
        predict, conf = self.model.predict_y(Xnew=tf.convert_to_tensor(value=X_in.astype(float), dtype=tf.float64))
        conf = conf.numpy()
        self.prediction = predict.numpy()
        if hasattr(self, 'standardize_y') and self.standardize_y:
            self.prediction = self.y_scaler.inverse_transform(predict)
            conf = self.y_scaler.inverse_transform(conf)
        return self.prediction.flatten(), self.var.flatten(), conf[:, 0]

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        # train model
        self.prediction = None
        self.retrain(retrain=train)
        # validate model
        return self.predict(X_in=val)
