import abc
import optuna
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


class BaseModel(abc.ABC):
    """
    BaseModel parent class for all models that can be used within the framework.

    Every model must be based on :obj:`~ForeTiS.model._base_model.BaseModel` directly or
    BaseModel's child classes, e.g. :obj:`~ForeTiS.model._sklearn_model.SklearnModel` or
    :obj:`~ForeTiS.model._torch_model.TorchModel`

    ** Attributes **

        * Instance attributes *

        - optuna_trial (*optuna.trial.Trial*): trial of optuna for optimization
        - datasets (*list<pd.DataFrame>*): all datasets that are available
        - n_outputs (*int*): number of outputs of the prediction model
        - all_hyperparams (*dict*): dictionary with all hyperparameters with related info that can be tuned (structure see :obj:`~ForeTiS.model._base_model.BaseModel.define_hyperparams_to_tune`)
        - dataset (*pd.DataFrame*): the dataset for this optimization trial
        - model: model object

    :param optuna_trial: Trial of optuna for optimization
    :param datasets: all datasets that are available
    :param featureset: on which featuresets the models should be optimized
    :param test_set_size_percentage: the size of the test set in percentage
    :param target_column: the target column for the prediction

    """

    # Constructor super class #
    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset: str,
                 test_set_size_percentage: int, target_column: str):
        self.optuna_trial = optuna_trial
        self.datasets = datasets
        self.target_column = target_column
        self.n_outputs = 1
        if not hasattr(self, 'all_hyperparams'):
            self.all_hyperparams = self.define_hyperparams_to_tune()
        else:
            # update in case common hyperparams are already defined
            self.all_hyperparams.update(self.define_hyperparams_to_tune())
        self.all_hyperparams.update(self.dim_reduction())
        dim_reduction = self.suggest_hyperparam_to_optuna('pca')
        del self.all_hyperparams['pca']
        if featureset == 'optimize':
            self.all_hyperparams.update(self.dataset_hyperparam())
            dataset_name = self.suggest_hyperparam_to_optuna('dataset')
            del self.all_hyperparams['dataset']
            for dataset in datasets.datasets:
                if dataset.name == dataset_name:
                    self.dataset = dataset
                    break
        else:
            for dataset in datasets.datasets:
                if dataset.name == featureset:
                    self.dataset = dataset
                    break
        if dim_reduction:
            self.dataset = self.pca_transform_train_test(test_set_size_percentage=test_set_size_percentage)
        self.model = self.define_model()

    # Methods required by each child class #
    @abc.abstractmethod
    def define_model(self):
        """
        Method that defines the model that needs to be optimized.
        Hyperparams to tune have to be specified in all_hyperparams and suggested via suggest_hyperparam_to_optuna().
        The hyperparameters have to be included directly in the model definiton to be optimized.
        e.g. if you want to optimize the number of layers, do something like

            .. code-block:: python

                n_layers = self.suggest_hyperparam_to_optuna('n_layers') # same name in define_hyperparams_to_tune()
                for layer in n_layers:
                    do something

        Then the number of layers will be optimized by optuna.
        """

    @abc.abstractmethod
    def define_hyperparams_to_tune(self) -> dict:
        """
        Method that defines the hyperparameters that should be tuned during optimization and their ranges.
        Required format is a dictionary with:

        .. code-block:: python

            {
                'name_hyperparam_1':
                    {
                    # MANDATORY ITEMS
                    'datatype': 'float' | 'int' | 'categorical',
                    FOR DATATYPE 'categorical':
                        'list_of_values': []  # List of all possible values
                    FOR DATATYPE ['float', 'int']:
                        'lower_bound': value_lower_bound,
                        'upper_bound': value_upper_bound,
                        # OPTIONAL ITEMS (only for ['float', 'int']):
                        'log': True | False  # sample value from log domain or not
                        'step': step_size # step of discretization.
                                            # Caution: cannot be combined with log=True
                                                            # - in case of 'float' in general and
                                                            # - for step!=1 in case of 'int'
                    },
                'name_hyperparam_2':
                    {
                    ...
                    },
                ...
                'name_hyperparam_k':
                    {
                    ...
                    }
            }

        If you want to use a similar hyperparameter multiple times (e.g. Dropout after several layers),
        you only need to specify the hyperparameter once. Individual parameters for every suggestion will be created.
        """

    @abc.abstractmethod
    def retrain(self, retrain: pd.DataFrame):
        """
        Method that runs the retraining of the model

        :param retrain: data for retraining
        """

    @abc.abstractmethod
    def update(self, update: pd.DataFrame, period: int):
        """
        Method that runs the updating of the model

        :param update: data for updating
        """

    @abc.abstractmethod
    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Method that predicts target values based on the input X_in

        :param X_in: feature matrix as input

        :return: numpy array with the predicted values
        """

    @abc.abstractmethod
    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Method that runs the whole training and validation loop

        :param train: data for the training
        :param val: data for validation

        :return: predictions on validation set
        """

    ### General methods ###
    def suggest_hyperparam_to_optuna(self, hyperparam_name: str):
        """
        Suggest a hyperparameter of hyperparam_dict to the optuna trial to optimize it.

        If you want to add a parameter to your model / in your pipeline to be optimized, you need to call this method

        :param hyperparam_name: name of the hyperparameter to be tuned (see :obj:`~ForeTiS.model._base_model.BaseModel.define_hyperparams_to_tune`)

        :return: suggested value
        """
        # Get specification of the hyperparameter
        if hyperparam_name in self.all_hyperparams:
            spec = self.all_hyperparams[hyperparam_name]
        else:
            raise Exception(hyperparam_name + ' not found in all_hyperparams dictionary.')

        # Check if the hyperparameter already exists in the trial and needs a suffix
        # (e.g. same dropout specification for multiple layers that should be optimized individually)
        if hyperparam_name in self.optuna_trial.params:
            counter = 1
            while True:
                current_name = hyperparam_name + '_' + str(counter)
                if current_name not in self.optuna_trial.params:
                    optuna_param_name = current_name
                    break
                counter += 1
        else:
            optuna_param_name = hyperparam_name

        # Read dict with specification for the hyperparamater and suggest it to the trial
        if spec['datatype'] == 'categorical':
            if 'list_of_values' not in spec:
                raise Exception(
                    '"list of values" for ' + hyperparam_name + ' not in hyperparams_dict. '
                    'Check define_hyperparams_to_tune() of the model.'
                )
            suggested_value = \
                self.optuna_trial.suggest_categorical(name=optuna_param_name, choices=spec['list_of_values'])
        elif spec['datatype'] in ['float', 'int']:
            if 'step' in spec:
                step = spec['step']
            else:
                step = None if spec['datatype'] == 'float' else 1
            log = spec['log'] if 'log' in spec else False
            if 'lower_bound' not in spec or 'upper_bound' not in spec:
                raise Exception(
                    '"lower_bound" or "upper_bound" for ' + hyperparam_name + ' not in all_hyperparams. '
                    'Check define_hyperparams_to_tune() of the model.'
                )
            if spec['datatype'] == 'int':
                suggested_value = self.optuna_trial.suggest_int(
                    name=optuna_param_name, low=spec['lower_bound'], high=spec['upper_bound'], step=step, log=log
                )
            else:
                suggested_value = self.optuna_trial.suggest_float(
                    name=optuna_param_name, low=spec['lower_bound'], high=spec['upper_bound'], step=step, log=log
                )
        else:
            raise Exception(
                spec['datatype'] + ' is not a valid parameter. Check define_hyperparams_to_tune() of the model.'
            )
        return suggested_value

    def suggest_all_hyperparams_to_optuna(self) -> dict:
        """
        Some models accept a dictionary with the model parameters.
        This method suggests all hyperparameters in all_hyperparams and gives back a dictionary containing them.

        :return: dictionary with suggested hyperparameters
        """
        for param_name in self.all_hyperparams.keys():
            _ = self.suggest_hyperparam_to_optuna(param_name)
        return self.optuna_trial.params

    def dataset_hyperparam(self):
        return {
            'dataset': {
                'datatype': 'categorical',
                'list_of_values': ['dataset_weather', 'dataset_cal', 'dataset_sales', 'dataset_weather_sales',
                                   'dataset_weather_cal', 'dataset_cal_sales', 'dataset_full'] # , 'dataset_full_corr', 'dataset_sales_corr', 'dataset_weather_sales_corr', 'dataset_cal_sales_corr'
            }
        }

    def dim_reduction(self):
        return {
            'pca': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }

    def pca_transform_train_test(self, test_set_size_percentage: int) -> tuple:
        """
        Deliver PCA transformed train and test set
        :return: tuple of transformed train and test dataset
        """
        if test_set_size_percentage == 2021:
            test = self.dataset.loc['2021-01-01': '2021-12-31']
            train_val = pd.concat([self.dataset, test]).drop_duplicates(keep=False)
        else:
            train_val, test = train_test_split(self.dataset, test_size=test_set_size_percentage * 0.01, shuffle=False)
        scaler = sklearn.preprocessing.StandardScaler()
        train_val_stand = scaler.fit_transform(train_val.drop(self.target_column, axis=1))
        pca = sklearn.decomposition.PCA(0.95)
        train_val_transf = pca.fit_transform(train_val_stand)
        test_stand = scaler.transform(test.drop(self.target_column, axis=1))
        test_transf = pca.transform(test_stand)
        train_val_data = pd.DataFrame(data=train_val_transf,
                                      columns=['PC' + str(i) for i in range(train_val_transf.shape[1])],
                                      index=train_val.index)
        train_val_data[self.target_column] = train_val[self.target_column]
        test_data = pd.DataFrame(data=test_transf, columns=['PC' + str(i) for i in range(test_transf.shape[1])],
                                 index=test.index)
        test_data[self.target_column] = test[self.target_column]
        dataset = pd.concat([train_val_data, test_data])
        return dataset

    def save_model(self, path: str, filename: str):
        """
        Persist the whole model object on a hard drive
        (can be loaded with :obj:`~ForeTiS.model._model_functions.load_model`)

        :param path: path where the model will be saved
        :param filename: filename of the model
        """
        joblib.dump(self, path + filename, compress=3)
