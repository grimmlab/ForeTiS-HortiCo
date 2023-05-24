import gpflow
import itertools
import numpy as np
import sklearn

from . import _tensorflow_model
from gpflow.kernels import Matern52, White, RationalQuadratic, Periodic, \
    SquaredExponential, Polynomial

class Gpr(_tensorflow_model.TensorflowModel):
    """
    Implementation of a class for Gpr.

    See :obj:`~ForeTiSHortiCo-Hortico.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> gpflow.models.GPR:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.conf = True

        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        self.standardize_y = self.suggest_hyperparam_to_optuna('standardize_y')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
        if self.standardize_y:
            self.y_scaler = sklearn.preprocessing.StandardScaler()

        optimizer_dict = {'Scipy': gpflow.optimizers.Scipy()}
        optimizer_key = self.suggest_hyperparam_to_optuna('optimizer')
        self.optimizer = optimizer_dict[optimizer_key]

        mean_function_dict = {'Constant': gpflow.mean_functions.Constant(),
                              None: None}
        mean_function_key = self.suggest_hyperparam_to_optuna('mean_function')
        mean_function = mean_function_dict[mean_function_key]
        kernel_key = self.suggest_hyperparam_to_optuna('kernel')
        kernel = self.kernel_dict[kernel_key]
        noise_variance = self.suggest_hyperparam_to_optuna('noise_variance')

        return gpflow.models.GPR(data=(np.zeros((5, 1)), np.zeros((5, 1))), kernel=kernel,  mean_function=mean_function,
                                 noise_variance=noise_variance)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        kernels, self.kernel_dict = self.extend_kernel_combinations()
        return {
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': kernels,
            },
            'noise_variance': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 100,
                'log': True
            },
            'optimizer': {
                'datatype': 'categorical',
                'list_of_values': ['Scipy']
            },
            'mean_function': {
                'datatype': 'categorical',
                'list_of_values': [None, 'Constant']
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'standardize_y': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }

    def extend_kernel_combinations(self):
        """
        Function extending kernels list with combinations based on base_kernels
        """
        kernels = []
        base_kernels = ['SquaredExponential', 'Matern52', 'WhiteKernel', 'RationalQuadratic', 'Polynomial',
                        'PeriodicSquaredExponential', 'PeriodicMatern52', 'PeriodicRationalQuadratic']
        kernel_dict = {
            'SquaredExponential': SquaredExponential(),
            'WhiteKernel': White(),
            'Matern52': Matern52(),
            'RationalQuadratic': RationalQuadratic(),
            'Polynomial': Polynomial(),
            'PeriodicSquaredExponential': Periodic(SquaredExponential(), period=52),
            'PeriodicMatern52': Periodic(Matern52(), period=52),
            'PeriodicRationalQuadratic': Periodic(RationalQuadratic(), period=52)
        }
        kernels.extend(base_kernels)
        for el in list(itertools.combinations(*[base_kernels], r=2)):
            kernels.append(el[0] + '+' + el[1])
            kernel_dict[el[0] + '+' + el[1]] = kernel_dict[el[0]] + kernel_dict[el[1]]
            kernels.append(el[0] + '*' + el[1])
            kernel_dict[el[0] + '*' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[1]]
        for el in list(itertools.combinations(*[base_kernels], r=3)):
            kernels.append(el[0] + '+' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '+' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '*' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '+' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '+' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '*' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[2] + '+' + el[1])
            kernel_dict[el[0] + '*' + el[2] + '+' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[2]] + kernel_dict[
                el[1]]
        return kernels, kernel_dict
