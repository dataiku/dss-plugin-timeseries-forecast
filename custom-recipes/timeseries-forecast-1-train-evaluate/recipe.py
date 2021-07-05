# -*- coding: utf-8 -*-
from gluonts_forecasts.mxnet_utils import set_mxnet_context

from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from dku_io_utils.utils import set_column_description
from gluonts_forecasts.training_session import TrainingSession
from dku_io_utils.recipe_config_loading import load_training_config, get_models_parameters
from dku_io_utils.utils import write_to_folder
from dku_io_utils.config_handler import create_dku_config
from dku_io_utils.dku_file_manager import DkuFileManager

from gluonts_forecasts.model_handler import get_model_label
from dku_constants import ObjectType, RECIPE
from timeseries_preparation.preparation import TimeseriesPreparator
from safe_logger import SafeLogger
from time import perf_counter


def create_dku_file_manager():
    file_manager = DkuFileManager()
    file_manager.add_input_dataset("input_dataset")
    file_manager.add_output_folder("model_folder")
    file_manager.add_output_dataset("evaluation_dataset")
    file_manager.add_output_dataset("evaluation_forecasts_dataset", required=False)
    return file_manager


def run():
    logger = SafeLogger("Forecast plugin")
    session_name = datetime.utcnow().isoformat() + "Z"
    logger.info("Starting training session {}...".format(session_name))

    recipe_config = get_recipe_config()
    file_manager = create_dku_file_manager()
    dku_config = create_dku_config(RECIPE.TRAIN_EVALUATE, recipe_config, file_manager)

    mxnet_context = set_mxnet_context(dku_config.gpu_devices)

    models_parameters = get_models_parameters(
        recipe_config, is_training_multivariate=dku_config.is_training_multivariate
    )
    start = perf_counter()

    training_df = file_manager.input_dataset.get_dataframe()

    timeseries_preparator = TimeseriesPreparator(
        time_column_name=dku_config.time_column_name,
        frequency=dku_config.frequency,
        target_columns_names=dku_config.target_columns_names,
        timeseries_identifiers_names=dku_config.timeseries_identifiers_names,
        external_features_columns_names=dku_config.external_features_columns_names,
        max_timeseries_length=dku_config.max_timeseries_length,
    )

    training_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(training_df)

    training_session = TrainingSession(
        target_columns_names=dku_config.target_columns_names,
        time_column_name=dku_config.time_column_name,
        frequency=dku_config.frequency,
        epoch=dku_config.epoch,
        models_parameters=models_parameters,
        prediction_length=dku_config.prediction_length,
        training_df=training_df_prepared,
        make_forecasts=dku_config.make_forecasts,
        external_features_columns_names=dku_config.external_features_columns_names,
        timeseries_identifiers_names=dku_config.timeseries_identifiers_names,
        batch_size=dku_config.batch_size,
        user_num_batches_per_epoch=dku_config.num_batches_per_epoch,
        season_length=dku_config.season_length,
        mxnet_context=mxnet_context,
    )
    training_session.init(partition_root=dku_config.partition_root, session_name=session_name)

    training_session.create_gluon_datasets()

    training_session.instantiate_models()

    training_session.train_evaluate(retrain=(not dku_config.evaluation_only))

    logger.info("Completed training and evaluation of all models")

    if not dku_config.evaluation_only:

        model_folder = file_manager.model_folder

        metrics_path = "{}/metrics.csv".format(training_session.session_path)
        write_to_folder(training_session.get_metrics_df(), model_folder, metrics_path, ObjectType.CSV)

        gluon_train_dataset_path = "{}/gluon_train_dataset.pk.gz".format(training_session.session_path)
        write_to_folder(
            training_session.full_list_dataset, model_folder, gluon_train_dataset_path, ObjectType.PICKLE_GZ
        )

        for model in training_session.models:
            model_path = "{}/{}/model.pk.gz".format(training_session.session_path, get_model_label(model.model_name))
            write_to_folder(model.predictor, model_folder, model_path, ObjectType.PICKLE_GZ)

            parameters_path = "{}/{}/params.json".format(
                training_session.session_path, get_model_label(model.model_name)
            )
            write_to_folder(model.model_parameters, model_folder, parameters_path, ObjectType.JSON)

    logger.info("Completed training session {} in {:.2f} seconds".format(session_name, perf_counter() - start))

    file_manager.evaluation_dataset.write_with_schema(training_session.get_evaluation_metrics_df())
    evaluation_results_columns_descriptions = training_session.create_evaluation_results_columns_descriptions()
    set_column_description(file_manager.evaluation_dataset, evaluation_results_columns_descriptions)

    if dku_config.make_forecasts:
        evaluation_forecasts_df = training_session.get_evaluation_forecasts_df()
        file_manager.evaluation_forecasts_dataset.write_with_schema(evaluation_forecasts_df)

        evaluation_forecasts_columns_descriptions = training_session.create_evaluation_forecasts_column_description()
        set_column_description(file_manager.evaluation_forecasts_dataset, evaluation_forecasts_columns_descriptions)


run()
