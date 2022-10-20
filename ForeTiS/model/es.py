import statsmodels.tsa.api

from . import _stat_model


class Es(_stat_model.StatModel):
    """
    Implementation of a class for Exponential Smoothing (ES).
    See :obj:`~ForeTiS.model._base_model.BaseModel` and
    :obj:`~ForeTiS.model._stat_model.StatModel` for more information on the attributes.
    """

    def define_model(self) -> statsmodels.tsa.api.ExponentialSmoothing:
        """
        Definition of an ES model.
        """
        self.remove_bias = self.suggest_hyperparam_to_optuna('remove_bias')
        self.use_brute = self.suggest_hyperparam_to_optuna('use_brute')
        endog = self.dataset[self.target_column].copy()

        trend = self.suggest_hyperparam_to_optuna('trend')
        damped_trend = self.suggest_hyperparam_to_optuna('damped_trend')
        seasonal = self.suggest_hyperparam_to_optuna('seasonal')
        seasonal_periods = self.suggest_hyperparam_to_optuna('seasonal_periods')

        if endog.eq(0).any().any() and seasonal == 'mul':
            endog += 0.01
        endog.index.freq = endog.index.inferred_freq

        if trend is None:
            damped_trend = False

        return statsmodels.tsa.api.ExponentialSmoothing(endog=endog, trend=trend, damped_trend=damped_trend,
                                                        seasonal=seasonal, seasonal_periods=seasonal_periods)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        See :obj:`~ForeTiS.model._stat_model.StatModel` for more information on hyperparameters
        common for all torch models.
        """
        return {
            'trend': {
                'datatype': 'categorical',
                'list_of_values': ['add', 'mul', None]
            },
            'damped_trend': {
                'datatype': 'categorical',
                'list_of_values': [False, True]
            },
            'seasonal': {
                'datatype': 'categorical',
                'list_of_values': ['add', 'mul', None]
            },
            'seasonal_periods': {
                'datatype': 'categorical',
                'list_of_values': [None, 52, 104]
            },
            'use_brute': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'remove_bias': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }
