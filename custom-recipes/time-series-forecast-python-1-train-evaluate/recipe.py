# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from plugin_io_utils import get_models_parameters, set_column_description, assert_time_column_is_date, assert_continuous_time_column
from dku_timeseries.global_models import GlobalModels
from plugin_config_loading import load_training_config

config = get_recipe_config()

global_params = load_training_config(config)
version_name = datetime.utcnow().isoformat()+'Z'

models_parameters = get_models_parameters(config)

training_dataset = dataiku.Dataset(global_params['input_dataset_name'])
assert_time_column_is_date(training_dataset, global_params['time_column_name'])
# order of cols is important (for the predict recipe)
columns = [global_params['time_column_name']] + global_params['target_columns_names'] + global_params['external_feature_columns']
training_df = training_dataset.get_dataframe(columns=columns)
assert_continuous_time_column(
    training_df, global_params['time_column_name'], global_params['time_granularity_unit'], global_params['time_granularity_step'])

global_models = GlobalModels(
    target_columns_names=global_params['target_columns_names'],
    time_column_name=global_params['time_column_name'],
    frequency=global_params['frequency'],
    model_folder=global_params['model_folder'],
    epoch=global_params['epoch'],
    models_parameters=models_parameters,
    prediction_length=global_params['prediction_length'],
    training_df=training_df,
    make_forecasts=global_params['make_forecasts'],
    external_features_column_name=global_params['external_feature_columns']
)
global_models.init_all_models()

global_models.evaluate_all(global_params['evaluation_strategy'])

global_models.fit_all()

global_models.save_all(version_name=version_name)

metrics_df = global_models.get_metrics_df()
global_params['evaluation_dataset'].write_with_schema(metrics_df)

metrics_column_descriptions = global_models.create_metrics_column_description()
set_column_description(global_params['evaluation_dataset'], metrics_column_descriptions)

if global_params['make_forecasts']:
    evaluation_forecasts_df = global_models.get_evaluation_forecasts_df()
    global_params['evaluation_forecasts'].write_with_schema(evaluation_forecasts_df)

    evaluation_forecasts_column_descriptions = global_models.create_evaluation_forecasts_column_description()
    set_column_description(global_params['evaluation_forecasts'], evaluation_forecasts_column_descriptions)
