import pprint
import configparser

from ForeTiS.utils import helper_functions
from ForeTiS.preprocess import base_dataset
from ForeTiS.optimization import optuna_optim


def run(data_dir: str, save_dir: str = None, featuresets: list = None, datasplit: str = 'timeseries-cv',
        test_set_size_percentage=None, val_set_size_percentage: int = 20, n_splits: int = 4, test_year: int = None,
        windowsize_current_statistics: int = 4, windowsize_lagged_statistics: int = 4,  cyclic_encoding: bool = False,
        imputation_method: str = 'None', correlation_method: str = None, correlation_number: int = None,
        models: list = None, target_column: str = None, n_trials: int = 100, pca_transform: bool =False,
        save_final_model: bool = False, periodical_refit_cycles: list = None, refit_drops: int = 0, data: str = None,
        refit_window: int = 5, intermediate_results_interval: int = None, batch_size: int = 32, n_epochs: int = None):

    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == ['all'] else models
    featureset_overview = {}
    model_featureset_overview = {}
    config = configparser.ConfigParser()
    config.read('Config/dataset_specific_config.ini')
    datasets = base_dataset.Dataset(data_dir=data_dir, data=data, test_set_size_percentage=test_set_size_percentage,
                                    target_column=target_column, cyclic_encoding=cyclic_encoding,
                                    windowsize_current_statistics=windowsize_current_statistics,
                                    windowsize_lagged_statistics=windowsize_lagged_statistics,
                                    imputation_method=imputation_method, correlation_method=correlation_method,
                                    correlation_number=correlation_number, config=config, test_year=test_year)
    print('### Dataset is loaded ###')
    for current_model_name in models_to_optimize:
        for featureset in featuresets:
            optuna_run = optuna_optim.OptunaOptim(save_dir=save_dir, data=data, featureset=featureset,
                                                  datasplit=datasplit, target_column=target_column, n_trials=n_trials,
                                                  test_set_size_percentage=test_set_size_percentage, models=models,
                                                  val_set_size_percentage=val_set_size_percentage, n_splits=n_splits,
                                                  save_final_model=save_final_model, pca_transform=pca_transform,
                                                  periodical_refit_cycles=periodical_refit_cycles,
                                                  refit_drops=refit_drops, refit_window=refit_window,
                                                  intermediate_results_interval=intermediate_results_interval,
                                                  batch_size=batch_size, n_epochs=n_epochs, datasets=datasets,
                                                  current_model_name=current_model_name, config=config)
            print('### Starting Optuna Optimization for model ' + current_model_name + ' and featureset ' + featureset
                  + ' ###')
            overall_results = optuna_run.run_optuna_optimization
            print('### Finished Optuna Optimization for ' + current_model_name + ' and featureset ' + featureset
                  + ' ###')
            featureset_overview[featureset] = overall_results
            model_featureset_overview[current_model_name] = featureset_overview
    print('# Optimization runs done for models ' + str(models_to_optimize) + ' and ' + str(featuresets))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=5).pprint(model_featureset_overview)
