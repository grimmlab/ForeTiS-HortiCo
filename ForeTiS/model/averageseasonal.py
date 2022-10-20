from . import _baseline_model


class AverageSeasonal(_baseline_model.BaselineModel):
    """See BaseModel for more information on the parameters"""

    def define_model(self):
        """See BaseModel for more information"""
        self.window = self.datasets.seasonal_periods
        return AverageSeasonal

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
        return {}
