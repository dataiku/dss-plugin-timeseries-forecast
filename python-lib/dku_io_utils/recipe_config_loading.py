import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)
import re
from gluonts_forecasts.model_handler import list_available_models
from dku_io_utils.partitions_handling import get_folder_partition_root, check_only_one_read_partition
from constants import FORECASTING_STYLE_PRESELECTED_MODELS, GPU_CONFIGURATION
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


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

    model_folder_name = get_output_names_for_role("model_folder")[0]
    params["model_folder"] = dataiku.Folder(model_folder_name)
    params["partition_root"] = get_folder_partition_root(params["model_folder"])
    check_only_one_read_partition(params["partition_root"], params["training_dataset"])

    evaluation_dataset_name = get_output_names_for_role("evaluation_dataset")[0]
    params["evaluation_dataset"] = dataiku.Dataset(evaluation_dataset_name)

    params["make_forecasts"] = False
    evaluation_forecasts_dataset_names = get_output_names_for_role("evaluation_forecasts_dataset")
    if len(evaluation_forecasts_dataset_names) > 0:
        params["evaluation_forecasts_dataset"] = dataiku.Dataset(evaluation_forecasts_dataset_names[0])
        params["make_forecasts"] = True

    params["time_column_name"] = recipe_config.get("time_column")
    if params["time_column_name"] not in training_dataset_columns:
        raise PluginParamValidationError(f"Invalid time column selection: {params['time_column_name']}")

    params["target_columns_names"] = sanitize_column_list(recipe_config.get("target_columns"))
    if len(params["target_columns_names"]) == 0 or not all(column in training_dataset_columns for column in params["target_columns_names"]):
        raise PluginParamValidationError(f"Invalid target column(s) selection: {params['target_columns_names']}")
    params["target_columns_names"] = reorder_column_list(params["target_columns_names"], training_dataset_columns)

    long_format = recipe_config.get("additional_columns", False)
    if long_format:
        params["timeseries_identifiers_names"] = sanitize_column_list(recipe_config.get("timeseries_identifiers", []))
        if not all(column in training_dataset_columns for column in params["timeseries_identifiers_names"]):
            raise PluginParamValidationError(f"Invalid time series identifiers selection: {params['timeseries_identifiers_names']}")
    else:
        params["timeseries_identifiers_names"] = []

    params["is_training_multivariate"] = True if (len(params["target_columns_names"]) > 1) or (len(params["timeseries_identifiers_names"]) > 0) else False

    if long_format and len(params["timeseries_identifiers_names"]) == 0:
        raise PluginParamValidationError("Long format is activated but no time series identifiers have been provided")

    external_feature_activated = recipe_config.get("external_feature_activated", False)
    if external_feature_activated:
        params["external_features_columns_names"] = sanitize_column_list(recipe_config.get("external_feature_columns", []))
    else:
        params["external_features_columns_names"] = []
    if not all(column in training_dataset_columns for column in params["external_features_columns_names"]):
        raise PluginParamValidationError(f"Invalid external features selection: {params['external_features_columns_names']}")

    params["frequency_unit"] = recipe_config.get("frequency_unit")

    if params["frequency_unit"] not in ["W", "H", "min"]:
        params["frequency"] = params["frequency_unit"]
    else:
        if params["frequency_unit"] == "W":
            params["frequency"] = f"W-{recipe_config.get('frequency_end_of_week', 1)}"
        elif params["frequency_unit"] == "H":
            params["frequency"] = f"{recipe_config.get('frequency_step_hours', 1)}H"
        elif params["frequency_unit"] == "min":
            params["frequency"] = f"{recipe_config.get('frequency_step_minutes', 1)}min"

    params["prediction_length"] = recipe_config.get("prediction_length")
    if not params["prediction_length"]:
        raise PluginParamValidationError("Please specify forecasting horizon")

    params["forecasting_style"] = recipe_config.get("forecasting_style", "auto")
    params["epoch"] = recipe_config.get("epoch", 10)
    params["batch_size"] = recipe_config.get("batch_size", 32)

    params["auto_num_batches_per_epoch"] = recipe_config.get("auto_num_batches_per_epoch", True)
    if params["auto_num_batches_per_epoch"]:
        params["num_batches_per_epoch"] = -1
    else:
        params["num_batches_per_epoch"] = recipe_config.get("num_batches_per_epoch", 50)

    if params["num_batches_per_epoch"] == 0:
        raise PluginParamValidationError("Number of batches per epoch cannot be 0")

    # Overwrite values in case of autoML mode selected
    if params["forecasting_style"] == "auto":
        params["epoch"] = 10
        params["batch_size"] = 32
        params["num_batches_per_epoch"] = 50
    elif params["forecasting_style"] == "auto_performance":
        params["context_length"] = params["prediction_length"]
        params["epoch"] = 30 if params["is_training_multivariate"] else 10
        params["batch_size"] = 32
        params["num_batches_per_epoch"] = -1

    params["sampling_method"] = recipe_config.get("sampling_method", "last_records")
    params["max_timeseries_length"] = None
    if params["sampling_method"] == "last_records":
        params["max_timeseries_length"] = recipe_config.get("number_records", 10000)
        if params["max_timeseries_length"] < 4:
            raise PluginParamValidationError("Number of records must be higher than 4")

    params["use_gpu"] = recipe_config.get("use_gpu", False)
    if params["use_gpu"]:
        params["gpu_location"] = recipe_config.get("gpu_location", "local_gpu")
        if params["gpu_location"] == "local_gpu":
            gpu_devices = recipe_config.get("gpu_devices", [])
            params["gpu_devices"] = parse_gpu_devices(gpu_devices)
        else:
            params["gpu_devices"] = [GPU_CONFIGURATION.CONTAINER_GPU]
    else:
        params["gpu_devices"] = None

    params["evaluation_strategy"] = "split"
    params["evaluation_only"] = False

    printable_params = {param: value for param, value in params.items() if "dataset" not in param and "folder" not in param}
    logger.info(f"Recipe parameters: {printable_params}")
    return params


def load_predict_config():
    """Utility function to load, resolve and validate all predict recipe config into a clean `params` dictionary

    Returns:
        Dictionary of parameter names (key) and values
    """
    params = {}
    recipe_config = get_recipe_config()

    # model folder
    model_folder = dataiku.Folder(get_input_names_for_role("model_folder")[0])
    params["model_folder"] = model_folder
    params["partition_root"] = get_folder_partition_root(params["model_folder"], is_input=True)

    params["external_features_future_dataset"] = None
    external_features_future_dataset_names = get_input_names_for_role("external_features_future_dataset")
    if len(external_features_future_dataset_names) > 0:
        params["external_features_future_dataset"] = dataiku.Dataset(external_features_future_dataset_names[0])

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify Forecast dataset in the 'Input / Output' tab of the recipe")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
    check_only_one_read_partition(params["partition_root"], params["model_folder"])
    check_only_one_read_partition(params["partition_root"], params["external_features_future_dataset"])

    params["manual_selection"] = True if recipe_config.get("model_selection_mode") == "manual" else False

    params["performance_metric"] = recipe_config.get("performance_metric")
    params["selected_session"] = recipe_config.get("manually_selected_session", "latest_session")
    params["selected_model_label"] = recipe_config.get("manually_selected_model_label")

    params["prediction_length"] = recipe_config.get("prediction_length", -1)
    params["confidence_interval"] = recipe_config.get("confidence_interval", 95)
    params["quantiles"] = convert_confidence_interval_to_quantiles(params["confidence_interval"])
    params["include_history"] = recipe_config.get("include_history", False)

    printable_params = {param: value for param, value in params.items() if "dataset" not in param and "folder" not in param}
    logger.info(f"Recipe parameters: {printable_params}")
    return params


def get_models_parameters(config, is_training_multivariate=False):
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
        if is_activated(config, model, is_training_multivariate):
            model_presets = get_model_presets(config, model)
            if "prediction_length" in model_presets.get("kwargs", {}):
                raise ValueError("Keyword argument 'prediction_length' is not writable, please use the Forecasting horizon parameter")
            models_parameters.update({model: model_presets})
    if not models_parameters:
        raise PluginParamValidationError("Please select at least one model")
    logger.info(f"Model parameters: {models_parameters}")
    return models_parameters


def is_activated(config, model, is_training_multivariate=False):
    """Returns the activation status for a model according to the selected forcasting style (auto / auto_performance) or UX config otherwise.

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().
        model (str): Model name found in the UI.

    Returns:
        True if model is activated, else False.
    """
    forecasting_style = config.get("forecasting_style", "auto") + ("_multivariate" if is_training_multivariate else "_univariate")
    if forecasting_style in FORECASTING_STYLE_PRESELECTED_MODELS:
        preselected_models = FORECASTING_STYLE_PRESELECTED_MODELS.get(forecasting_style)
        return model in preselected_models
    return config.get(f"{model}_model_activated", False)


def get_model_presets(config, model):
    """Collect all the parameters model from the UI and return them as a dict.

    Args:
        config (dict): Recipe config dictionary obtained with dataiku.customrecipe.get_recipe_config().
        model (str): Model name found in the UI.

    Returns:
        Dictionary of model parameters to be used as kwargs in gluonts Predictor.
    """
    model_presets = {}
    matching_key = f"{model}_model_(.*)"
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


def convert_confidence_interval_to_quantiles(confidence_interval):
    """Convert a confidence interval value into a list of lower and upper quantiles with also the median.

    Args:
        confidence_interval (int): Percentage.

    Raises:
        PluginParamValidationError: If the selected confidence interval is not between 1 and 99.

    Returns:
        List of quantiles.
    """
    if confidence_interval < 1 or confidence_interval > 99:
        raise PluginParamValidationError("Please choose a confidence interval between 1 and 99.")
    alpha = (100 - confidence_interval) / 2 / 100.0
    quantiles = [round(alpha, 3), 0.5, round(1 - alpha, 3)]
    return quantiles


def reorder_column_list(column_list_to_reorder, reference_column_list):
    """ Keep the target list in same order as the training dataset, for consistency of forecasted columns order"""
    reordered_list = []
    for column_name in reference_column_list:
        if column_name in column_list_to_reorder:
            reordered_list.append(column_name)
    return reordered_list


def parse_gpu_devices(gpu_devices):
    """Check the custom python MULTISELECT for GPU devices

    Args:
        gpu_devices (list): List of GPU number or ["no_gpu"]

    Raises:
        PluginParamValidationError:
            If more than 1 GPU are selected (for now we support only one GPU)
            If selected value is "no_gpu"

    Returns:
        List of a single GPU (we may support multiple later) or None
    """
    if len(gpu_devices) == 0:  # nothing selected
        raise PluginParamValidationError("Please select one local GPU device")
    elif len(gpu_devices) > 1:
        raise PluginParamValidationError("GluonTS does not currently support multi-GPU training. Please select only one GPU device.")
    else:  # one element list
        if gpu_devices[0] == GPU_CONFIGURATION.NO_GPU:
            raise PluginParamValidationError("No GPU available, please check your CUDA installation or deactivate the 'Use GPU' parameter")
        else:
            return [int(gpu_device.split("_")[1]) for gpu_device in gpu_devices]
