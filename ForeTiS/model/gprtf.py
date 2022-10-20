import gpflow
import itertools
import numpy as np

from . import _tensorflow_model
from gpflow.kernels import Matern52, White, RationalQuadratic, Periodic, \
    SquaredExponential, Polynomial

class Gpr(_tensorflow_model.TensorflowModel):
    """See BaseModel for more information on the parameters"""

    def define_model(self) -> gpflow.models.GPR:
        """See BaseModel for more information"""
        self.variance = True

        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        self.standardize_y = self.suggest_hyperparam_to_optuna('standardize_y')

        optimizer_dict = {'Scipy': gpflow.optimizers.Scipy()}
        optimizer_key = self.suggest_hyperparam_to_optuna('optimizer')
        self.optimizer = optimizer_dict[optimizer_key]

        mean_function_dict = {'Constant': gpflow.mean_functions.Constant(),
                              'None': None}
        mean_function_key = self.suggest_hyperparam_to_optuna('mean_function')
        mean_function = mean_function_dict[mean_function_key]
        kernel_key = self.suggest_hyperparam_to_optuna('kernel')
        kernel = self.kernel_dict[kernel_key]
        noise_variance = self.suggest_hyperparam_to_optuna('noise_variance')

        return gpflow.models.GPR(data=(np.zeros((5, 1)), np.zeros((5, 1))), kernel=kernel,  mean_function=mean_function,
                                 noise_variance=noise_variance)

    def define_hyperparams_to_tune(self) -> dict:
        """See BaseModel for more information on the format"""
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
                'list_of_values': ['None', 'Constant']
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

    # 'kernel': {
    #     'datatype': 'categorical',
    #     'list_of_values': ['SquaredExponential+Matern*PeriodicRationalQuadratic'],
    # },
    # 'noise_variance': {
    #     'datatype': 'float',
    #     'lower_bound': 10,
    #     'upper_bound': 10,
    # },
    # 'optimizer': {
    #     'datatype': 'categorical',
    #     'list_of_values': ['Scipy']
    # },
    # 'mean_function': {
    #     'datatype': 'categorical',
    #     'list_of_values': ['Constant']
    # },
    # 'standardize_X': {
    #     'datatype': 'categorical',
    #     'list_of_values': [False]
    # },
    # 'standardize_y': {
    #     'datatype': 'categorical',
    #     'list_of_values': [False]
    # }

    def extend_kernel_combinations(self):
        """
        Function extending kernels list with combinations based on base_kernels
        """
        kernels = []
        base_kernels = ['SquaredExponential', 'Matern', 'WhiteKernel', 'RationalQuadratic', 'Polynomial',
                        'PeriodicSquaredExponential', 'PeriodicMatern52', 'PeriodicRationalQuadratic']
        kernel_dict = {
            'SquaredExponential': SquaredExponential(),
            'WhiteKernel': White(),
            'Matern': Matern52(),
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