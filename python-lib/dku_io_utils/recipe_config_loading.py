import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)
import re
from gluonts_forecasts.model_handler import list_available_models
from dku_io_utils.utils import get_partition_root
from constants import FORECASTING_STYLE_PRESELECTED_MODELS


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


def load_training_config(recipe_config):
    """Utility function to load, resolve and validate all training recipe config into a clean `params` dictionary

    Returns:
        Dictionary of parameter names (key) and values
    """
    params = {}

    input_dataset_name = get_input_names_for_role("input_dataset")[0]
    params["training_dataset"] = dataiku.Dataset(input_dataset_name)
    training_dataset_columns = [p["name"] for p in params["training_dataset"].read_schema()]
    params["partition_root"] = get_partition_root(params["training_dataset"])

    model_folder_name = get_output_names_for_role("model_folder")[0]
    params["model_folder"] = dataiku.Folder(model_folder_name)

    evaluation_dataset_name = get_output_names_for_role("evaluation_dataset")[0]
    params["evaluation_dataset"] = dataiku.Dataset(evaluation_dataset_name)

    params["make_forecasts"] = False
    evaluation_forecasts_names = get_output_names_for_role("evaluation_forecasts")
    if len(evaluation_forecasts_names) > 0:
        params["evaluation_forecasts"] = dataiku.Dataset(evaluation_forecasts_names[0])
        params["make_forecasts"] = True

    params["time_column_name"] = recipe_config.get("time_column")
    if params["time_column_name"] not in training_dataset_columns:
        raise PluginParamValidationError("Invalid time column selection")

    params["target_columns_names"] = sanitize_column_list(recipe_config.get("target_columns"))
    if len(params["target_columns_names"]) == 0 or not all(column in training_dataset_columns for column in params["target_columns_names"]):
        raise PluginParamValidationError("Invalid target column(s) selection")

    long_format = recipe_config.get("additional_columns", False)
    if long_format:
        params["timeseries_identifiers_names"] = sanitize_column_list(recipe_config.get("timeseries_identifiers", []))
        if not all(column in training_dataset_columns for column in params["timeseries_identifiers_names"]):
            raise PluginParamValidationError("Invalid timeseries identifiers column(s) selection")
    else:
        params["timeseries_identifiers_names"] = []

    if long_format and len(params["timeseries_identifiers_names"]) == 0:
        raise PluginParamValidationError("Long format is activated but no time series idenfiers are selected")

    params["external_features_columns_names"] = sanitize_column_list(recipe_config.get("external_feature_columns", []))
    if not all(column in training_dataset_columns for column in params["external_features_columns_names"]):
        raise PluginParamValidationError("Invalid external features column(s) selection")

    params["frequency_unit"] = recipe_config.get("frequency_unit")

    if params["frequency_unit"] not in ["H", "min"]:
        params["frequency"] = params["frequency_unit"]
    else:
        if params["frequency_unit"] == "H":
            params["frequency_step"] = recipe_config.get("frequency_step_hours", 1)
        elif params["frequency_unit"] == "min":
            params["frequency_step"] = recipe_config.get("frequency_step_minutes", 1)
        params["frequency"] = "{}{}".format(params["frequency_step"], params["frequency_unit"])

    params["prediction_length"] = recipe_config.get("prediction_length")
    if params["prediction_length"] is None:
        raise PluginParamValidationError("Prediction length is not set.")

    params["context_length"] = recipe_config.get("context_length", -1)
    if params["context_length"] == -1:
        params["context_length"] = params["prediction_length"]

    params["forecasting_style"] = recipe_config.get("forecasting_style", "auto")
    if params["forecasting_style"] == "auto":
        params["epoch"] = 50
    elif params["forecasting_style"] == "auto_performance":
        params["epoch"] = 100
    else:
        params["epoch"] = recipe_config.get("epoch", 10)
    params["batch_size"] = recipe_config.get("batch_size", 32)
    params["num_batches_per_epoch"] = recipe_config.get("num_batches_per_epoch", 50)

    # V2 implement
    params["gpu"] = recipe_config.get("gpu", "no_gpu")
    params["evaluation_strategy"] = "split"
    params["evaluation_only"] = False

    return params


def load_predict_config():
    """Utility function to load, resolve and validate all predict recipe config into a clean `params` dictionary

    Returns:
        Dictionary of parameter names (key) and values
    """
    params = {}
    recipe_config = get_recipe_config()

    # input folder
    model_folder = dataiku.Folder(get_input_names_for_role("model_folder")[0])
    params["model_folder"] = model_folder

    params["external_features_future_dataset"] = None
    external_features_future_dataset_names = get_input_names_for_role("external_features_future_dataset")
    if len(external_features_future_dataset_names) > 0:
        params["external_features_future_dataset"] = dataiku.Dataset(external_features_future_dataset_names[0])

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
    params["partition_root"] = get_partition_root(params["output_dataset"])

    params["manual_selection"] = True if recipe_config.get("model_selection_mode") == "manual" else False

    params["performance_metric"] = recipe_config.get("performance_metric")
    params["selected_session"] = recipe_config.get("manually_selected_session")
    params["selected_model_label"] = recipe_config.get("manually_selected_model_label")

    params["prediction_length"] = recipe_config.get("prediction_length")
    params["quantiles"] = recipe_config.get("quantiles")
    if any(x < 0 or x > 1 for x in params["quantiles"]):
        raise PluginParamValidationError("Quantiles must be between 0 and 1.")
    params["quantiles"].sort()

    params["include_history"] = recipe_config.get("include_history")

    return params


def get_models_parameters(config):
    """Create a models parameters dictionary to store for each activated model its parameters (activated, kwargs, ...)

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().

    Raises:
        ValueError: If a prediction_length parameter is trying to be set in the model params.

    Returns:
        Dictionary of model parameter (value) by activated model name (key).
    """
    models_parameters = {}
    for model in list_available_models():
        if is_activated(config, model):
            model_presets = get_model_presets(config, model)
            if "prediction_length" in model_presets.get("kwargs", {}):
                raise ValueError("The value for 'prediction_length' cannot be changed")
            models_parameters.update({model: model_presets})
    models_parameters = set_naive_model_parameters(config, models_parameters)
    return models_parameters


def set_naive_model_parameters(config, models_parameters):
    """Update models_parameters to add specific parameters that some baselines models have.

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().
        models_parameters (dict): Obtained with get_models_parameters.

    Returns:
        Dictionary of model parameter (value) by activated model name (key).
    """
    naive_model_parameters = models_parameters.get("naive")
    if naive_model_parameters is not None:
        model_name = config.get("naive_model_method")
        models_parameters[model_name] = models_parameters.pop("naive")
        if model_name in ["trivial_identity", "trivial_mean"]:
            models_parameters[model_name]["kwargs"] = {"num_samples": 100}
    return models_parameters


def is_activated(config, model):
    """Returns the activation status for a model according to the selected forcasting style (auto / auto_performance) or UX config otherwise.

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().
        model (str): Model name found in the UI.

    Returns:
        True if model is activated, else False.
    """
    forecasting_style = config.get("forecasting_style", "auto")
    if forecasting_style in FORECASTING_STYLE_PRESELECTED_MODELS:
        preselected_models = FORECASTING_STYLE_PRESELECTED_MODELS.get(forecasting_style)
        return model in preselected_models
    return config.get("{}_model_activated".format(model), False)


def get_model_presets(config, model):
    """Collect all the parameters model from the UI and return them as a dict.

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().
        model (str): Model name found in the UI.

    Returns:
        Dictionary of model parameters to be used as kwargs in gluonts Predictor.
    """
    model_presets = {}
    matching_key = "{}_model_(.*)".format(model)
    for key in config:
        key_search = re.match(matching_key, key, re.IGNORECASE)
        if key_search:
            key_type = key_search.group(1)
            model_presets.update({key_type: config[key]})
    return model_presets


def sanitize_column_list(input_column_list):
    """ Remove empty elements (Nones, '') from input columns list"""
    sanitized_column_list = [input for input in input_column_list if input]
    return sanitized_column_list
