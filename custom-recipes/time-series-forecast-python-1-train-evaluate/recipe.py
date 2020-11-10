# -*- coding: utf-8 -*-
from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from dku_io_utils.dku_io_utils import get_models_parameters, set_column_description
from gluonts_forecasts.global_models import GlobalModels
from dku_io_utils.dku_config_loading import load_training_config

config = get_recipe_config()
params = load_training_config(config)
version_name = datetime.utcnow().isoformat()+'Z'

models_parameters = get_models_parameters(config)

training_df = params['training_dataset'].get_dataframe(columns=params['columns_to_keep'])

global_models = GlobalModels(
    target_columns_names=params['target_columns_names'],
    time_column_name=params['time_column_name'],
    frequency=params['frequency'],
    epoch=params['epoch'],
    models_parameters=models_parameters,
    prediction_length=params['prediction_length'],
    training_df=training_df,
    make_forecasts=params['make_forecasts'],
    external_features_columns_names=params['external_features_columns_names'],
    timeseries_identifiers_names=params['timeseries_identifiers_names'],
    batch_size=params['batch_size'],
    gpu=params['gpu'],
    context_length=params['context_length']
)
global_models.init_all_models(partition_root=params['partition_root'], version_name=version_name)

global_models.evaluate_all(params['evaluation_strategy'])

if not params['evaluation_only']:
    global_models.fit_all()
    global_models.save_all(model_folder=params['model_folder'])

metrics_df = global_models.get_metrics_df()

params['evaluation_dataset'].write_with_schema(metrics_df)

metrics_column_descriptions = global_models.create_metrics_column_description()
set_column_description(params['evaluation_dataset'], metrics_column_descriptions)

if params['make_forecasts']:
    evaluation_forecasts_df = global_models.get_evaluation_forecasts_df()
    params['evaluation_forecasts'].write_with_schema(evaluation_forecasts_df)

    evaluation_forecasts_column_descriptions = global_models.create_evaluation_forecasts_column_description()
    set_column_description(params['evaluation_forecasts'], evaluation_forecasts_column_descriptions)
