import torch
import optuna
import sklearn
import pandas as pd
import numpy as np

from . import _torch_model
from ._model_classes import GetOutputZero, PrepareForlstm, PrepareForDropout
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


class LSTM(_torch_model.TorchModel):
    """
    Implementation of a class for a Long Short-Term Memory (LSTM) network.
    See :obj:`~ForeTiS.model._base_model.BaseModel` and
    :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on the attributes.
    """

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of a LSTM network.
        Architecture:
            - LSTM, Dropout, Linear
            - Linear output layer
        Number of output channels of the first layer, dropout rate, frequency of a doubling of the output channels and
        number of units in the first linear layer. may be fixed or optimized.
        """
        self.variance = True
        self.y_scaler = sklearn.preprocessing.StandardScaler()
        self.sequential = True
        self.seq_length = self.suggest_hyperparam_to_optuna('seq_length')
        model = []
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        p = self.suggest_hyperparam_to_optuna('dropout')
        n_feature = self.dataset.shape[1]
        lstm_hidden_dim = self.suggest_hyperparam_to_optuna('lstm_hidden_dim')

        model.append(PrepareForlstm())
        model.append(torch.nn.LSTM(input_size=n_feature, hidden_size=lstm_hidden_dim, num_layers=n_layers,
                                   dropout=p))
        model.append(GetOutputZero())
        model.append(PrepareForDropout())
        model.append(torch.nn.Dropout(p))
        model.append(torch.nn.Linear(in_features=lstm_hidden_dim, out_features=self.n_outputs))

        bnn_prior_parameters = {
            "prior_mu": self.suggest_hyperparam_to_optuna('prior_mu'),
            "prior_sigma": self.suggest_hyperparam_to_optuna('prior_sigma'),
            "posterior_mu_init": self.suggest_hyperparam_to_optuna('posterior_mu_init'),
            "posterior_rho_init": self.suggest_hyperparam_to_optuna('posterior_rho_init'),
            "type": self.suggest_hyperparam_to_optuna('type'),
            "moped_enable": self.suggest_hyperparam_to_optuna('moped_enable')
        }
        model = torch.nn.Sequential(*model)
        dnn_to_bnn(model, bnn_prior_parameters)
        return model


    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        See :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on hyperparameters
        common for all torch models.
        """
        return {
            'lstm_hidden_dim': {
                'datatype': 'int',
                'lower_bound': 5,
                'upper_bound': 100
            },
            'seq_length': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 104
            },
            'prior_mu': {
                'datatype': 'int',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'prior_sigma': {
                'datatype': 'int',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'posterior_mu_init': {
                'datatype': 'int',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'posterior_rho_init': {
                'datatype': 'int',
                'lower_bound': -3.0,
                'upper_bound': 3.0
            },
            'type': {
                'datatype': 'categorical',
                'list_of_values': ['Flipout', 'Reparameterization']
            },
            'moped_enable': {
                'datatype': 'categorical',
                'list_of_values': [False]
            },
            'num_monte_carlo': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
            }
        }

    def train_val_loader(self, train: pd.DataFrame, val: pd.DataFrame):
        train_loader = self.get_dataloader(X=train.drop(labels=[self.target_column], axis=1),
                                           y=train[self.target_column], only_transform=False)
        val_loader = self.get_dataloader(X=pd.concat([train.tail(self.seq_length), val]).
                                         drop(labels=[self.target_column], axis=1),
                                         y=pd.concat([train.tail(self.seq_length), val])[self.target_column],
                                         only_transform=True)
        val = pd.concat([train.tail(self.seq_length), val])
        return train_loader, val_loader, val

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        self.model.eval()
        predictions = None
        if self.variance:
            var = None
        if type(X_in) == pd.DataFrame:
            dataloader = self.get_dataloader(X=X_in.drop(labels=[self.target_column], axis=1),
                                             y=X_in[self.target_column], only_transform=True, predict=True)
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs.view(1, self.seq_length, -1)
                    inputs = inputs.to(device=self.device)
                    if self.variance:
                        predictions_mc = []
                        for _ in range(self.num_monte_carlo):
                            output = self.model(inputs)
                            predictions_mc.append(output)
                        predictions_ = torch.stack(predictions_mc)
                        outputs = torch.mean(predictions_, dim=0)
                        variance = torch.var(predictions_, dim=0)
                    else:
                        outputs = self.model(inputs)
                    predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
                    if self.variance:
                        var = torch.clone(variance) if var is None else torch.cat((var, variance))
        else:
            inputs = X_in.reshape(1, self.seq_length, -1)
            with torch.no_grad():
                inputs = torch.tensor(inputs.astype(np.float32))
                inputs = inputs.to(device=self.device)
                if self.variance:
                    predictions_mc = []
                    for _ in range(self.num_monte_carlo):
                        output = self.model(inputs)
                        predictions_mc.append(output)
                    predictions_ = torch.stack(predictions_mc)
                    outputs = torch.mean(predictions_, dim=0)
                    variance = torch.var(predictions_, dim=0)
                else:
                    outputs = self.model(inputs)
                predictions = torch.clone(outputs)
                if self.variance:
                    var = torch.clone(variance)
        self.prediction = self.y_scaler.inverse_transform(predictions.cpu().detach().numpy()).flatten()
        if self.variance:
            var = self.y_scaler.inverse_transform(var)
            return self.prediction, self.var_artifical, var.flatten()
        else:
            return self.prediction, self.var_artifical





