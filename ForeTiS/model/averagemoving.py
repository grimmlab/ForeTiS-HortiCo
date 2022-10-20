from . import _baseline_model


class AverageMoving(_baseline_model.BaselineModel):
    """See BaseModel for more information on the parameters"""

    def define_model(self):
        """See BaseModel for more information"""
        self.window = self.suggest_hyperparam_to_optuna('window')
        return AverageMoving

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {
            'window': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 20
            }
        }
