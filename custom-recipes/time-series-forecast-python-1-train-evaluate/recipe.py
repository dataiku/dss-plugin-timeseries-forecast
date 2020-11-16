# -*- coding: utf-8 -*-
from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from dku_io_utils.utils import get_models_parameters, set_column_description
from gluonts_forecasts.training_session import TrainingSession
from dku_io_utils.recipe_config_loading import load_training_config
from dku_io_utils.utils import write_to_folder

config = get_recipe_config()
params = load_training_config(config)
version_name = datetime.utcnow().isoformat() + "Z"

models_parameters = get_models_parameters(config)

training_df = params["training_dataset"].get_dataframe(columns=params["columns_to_keep"])

training_session = TrainingSession(
    target_columns_names=params["target_columns_names"],
    time_column_name=params["time_column_name"],
    frequency=params["frequency"],
    epoch=params["epoch"],
    models_parameters=models_parameters,
    prediction_length=params["prediction_length"],
    training_df=training_df,
    make_forecasts=params["make_forecasts"],
    external_features_columns_names=params["external_features_columns_names"],
    timeseries_identifiers_names=params["timeseries_identifiers_names"],
    batch_size=params["batch_size"],
    gpu=params["gpu"],
    context_length=params["context_length"],
)
training_session.init(partition_root=params["partition_root"], version_name=version_name)

training_session.evaluate(params["evaluation_strategy"])

metrics_df = training_session.get_metrics_df()
params["evaluation_dataset"].write_with_schema(metrics_df)

if not params["evaluation_only"]:
    training_session.train()
    # training_session.save(model_folder=params["model_folder"])

    model_folder = params["model_folder"]

    metrics_path = "{}/metrics.csv".format(version_name)
    write_to_folder(metrics_df, model_folder, metrics_path, "csv")

    gluon_train_dataset_path = "{}/gluon_train_dataset.pk.gz".format(version_name)
    write_to_folder(training_session.test_list_dataset, model_folder, gluon_train_dataset_path, "pickle.gz")

    for model in training_session.models:
        model_path = "{}/{}/model.pk.gz".format(version_name, model.model_name)
        write_to_folder(model.predictor, model_folder, model_path, "pickle.gz")

        parameters_path = "{}/{}/params.json".format(version_name, model.model_name)
        write_to_folder(model.model_parameters, model_folder, parameters_path, "json")


metrics_column_descriptions = training_session.create_metrics_column_description()
set_column_description(params["evaluation_dataset"], metrics_column_descriptions)

if params["make_forecasts"]:
    evaluation_forecasts_df = training_session.get_evaluation_forecasts_df()
    params["evaluation_forecasts"].write_with_schema(evaluation_forecasts_df)

    evaluation_forecasts_column_descriptions = training_session.create_evaluation_forecasts_column_description()
    set_column_description(params["evaluation_forecasts"], evaluation_forecasts_column_descriptions)
