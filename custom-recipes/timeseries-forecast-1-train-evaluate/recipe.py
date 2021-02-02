# -*- coding: utf-8 -*-
from gluonts_forecasts.mxnet_utils import set_mxnet_context

from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from dku_io_utils.utils import set_column_description
from gluonts_forecasts.training_session import TrainingSession
from dku_io_utils.recipe_config_loading import load_training_config, get_models_parameters
from dku_io_utils.utils import write_to_folder
from gluonts_forecasts.model_handler import get_model_label
from constants import ObjectType
from timeseries_preparation.preparation import TimeseriesPreparator
from safe_logger import SafeLogger
from time import perf_counter

logger = SafeLogger("Forecast plugin")
session_name = datetime.utcnow().isoformat() + "Z"
logger.info("Starting training session {}...".format(session_name))

config = get_recipe_config()
params = load_training_config(config)

mxnet_context = set_mxnet_context(params["gpu_devices"])

models_parameters = get_models_parameters(config, is_training_multivariate=params["is_training_multivariate"])
start = perf_counter()

training_df = params["training_dataset"].get_dataframe()

timeseries_preparator = TimeseriesPreparator(
    time_column_name=params["time_column_name"],
    frequency=params["frequency"],
    target_columns_names=params["target_columns_names"],
    timeseries_identifiers_names=params["timeseries_identifiers_names"],
    external_features_columns_names=params["external_features_columns_names"],
    max_timeseries_length=params["max_timeseries_length"],
)

training_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(training_df)

training_session = TrainingSession(
    target_columns_names=params["target_columns_names"],
    time_column_name=params["time_column_name"],
    frequency=params["frequency"],
    epoch=params["epoch"],
    models_parameters=models_parameters,
    prediction_length=params["prediction_length"],
    training_df=training_df_prepared,
    make_forecasts=params["make_forecasts"],
    external_features_columns_names=params["external_features_columns_names"],
    timeseries_identifiers_names=params["timeseries_identifiers_names"],
    batch_size=params["batch_size"],
    user_num_batches_per_epoch=params["num_batches_per_epoch"],
    season_length=params["season_length"],
    mxnet_context=mxnet_context,
)
training_session.init(partition_root=params["partition_root"], session_name=session_name)

training_session.create_gluon_datasets()

training_session.instantiate_models()

training_session.evaluate()

logger.info("Completed evaluation of all models")

if not params["evaluation_only"]:
    logger.info("Re-training models on entire dataset")
    training_session.train()

    model_folder = params["model_folder"]

    metrics_path = "{}/metrics.csv".format(training_session.session_path)
    write_to_folder(training_session.get_metrics_df(), model_folder, metrics_path, ObjectType.CSV)

    gluon_train_dataset_path = "{}/gluon_train_dataset.pk.gz".format(training_session.session_path)
    write_to_folder(training_session.full_list_dataset, model_folder, gluon_train_dataset_path, ObjectType.PICKLE_GZ)

    for model in training_session.models:
        model_path = "{}/{}/model.pk.gz".format(training_session.session_path, get_model_label(model.model_name))
        write_to_folder(model.predictor, model_folder, model_path, ObjectType.PICKLE_GZ)

        parameters_path = "{}/{}/params.json".format(training_session.session_path, get_model_label(model.model_name))
        write_to_folder(model.model_parameters, model_folder, parameters_path, ObjectType.JSON)

logger.info("Completed training session {} in {:.2f} seconds".format(session_name, perf_counter() - start))

params["evaluation_dataset"].write_with_schema(training_session.get_evaluation_metrics_df())
evaluation_results_columns_descriptions = training_session.create_evaluation_results_columns_descriptions()
set_column_description(params["evaluation_dataset"], evaluation_results_columns_descriptions)

if params["make_forecasts"]:
    evaluation_forecasts_df = training_session.get_evaluation_forecasts_df()
    params["evaluation_forecasts_dataset"].write_with_schema(evaluation_forecasts_df)

    evaluation_forecasts_columns_descriptions = training_session.create_evaluation_forecasts_column_description()
    set_column_description(params["evaluation_forecasts_dataset"], evaluation_forecasts_columns_descriptions)
