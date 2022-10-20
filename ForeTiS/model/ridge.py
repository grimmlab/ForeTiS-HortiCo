import sklearn

from . import _sklearn_model


class Ridge(_sklearn_model.SklearnModel):
    """See BaseModel for more information on the parameters"""

    def define_model(self) -> sklearn.linear_model.Ridge:
        """See BaseModel for more information"""
        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')

        alpha = self.suggest_hyperparam_to_optuna('alpha')
        params = {}
        params.update({'random_state': 42})
        params.update({'fit_intercept': True})
        params.update({'copy_X': True})
        params.update({'max_iter': None})
        params.update({'tol': 1e-3})
        params.update({'solver': 'auto'})
        return sklearn.linear_model.Ridge(alpha=alpha, **params)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'alpha': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }
