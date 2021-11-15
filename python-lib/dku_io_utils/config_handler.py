from dku_config import DkuConfig
from dku_constants import RECIPE
import re
from gluonts.time_feature import get_seasonality
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from dku_io_utils.partitions_handling import get_folder_partition_root, check_only_one_read_partition
from dku_constants import FORECASTING_STYLE_PRESELECTED_MODELS, GPU_CONFIGURATION, DEFAULT_SEASONALITIES
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


def create_dku_config(recipe_id, config, file_manager):
    dku_config = DkuConfig()
    if recipe_id == RECIPE.TRAIN_EVALUATE:
        add_train_evaluate_config(dku_config, config, file_manager)
    elif recipe_id == RECIPE.PREDICT:
        add_predict_config(dku_config, config, file_manager)

    logger.info(f"Created dku_config:\n{dku_config}")
    return dku_config


def add_train_evaluate_config(dku_config, config, file_manager):
    input_dataset_columns = get_column_names(file_manager.input_dataset)

    dku_config.add_param(name="partition_root", value=get_folder_partition_root(file_manager.model_folder))
    check_only_one_read_partition(dku_config.partition_root, file_manager.input_dataset)

    dku_config.add_param(
        name="make_forecasts",
        value=bool(file_manager.evaluation_forecasts_dataset is not None),
    )

    dku_config.add_param(
        name="time_column_name",
        label="Time column",
        value=config.get("time_column"),
        checks=[
            {"type": "is_type", "op": str},
            {
                "type": "in",
                "op": input_dataset_columns,
                "err_msg": "Invalid time column selection: {}.".format(config.get("time_column")),
            },
        ],
        required=True,
    )

    dku_config.add_param(
        name="target_columns_names",
        label="Target column(s)",
        value=sanitize_column_list(config.get("target_columns")),
        checks=[
            {"type": "is_type", "op": list},
            {
                "type": "is_subset",
                "op": input_dataset_columns,
                "err_msg": "Invalid target column(s) selection: {}.".format(config.get("target_columns_names")),
            },
        ],
        required=True,
    )
    dku_config.add_param(
        name="target_columns_names",
        value=reorder_column_list(dku_config.target_columns_names, input_dataset_columns),
    )

    long_format = config.get("additional_columns", False)
    if long_format:
        dku_config.add_param(
            name="timeseries_identifiers_names",
            value=sanitize_column_list(config.get("timeseries_identifiers", [])),
            checks=[
                {"type": "is_type", "op": list},
                {
                    "type": "is_subset",
                    "op": input_dataset_columns,
                    "err_msg": "Invalid time series identifiers selection: {}.".format(
                        config.get("timeseries_identifiers_names")
                    ),
                },
            ],
        )
    else:
        dku_config.add_param(name="timeseries_identifiers_names", value=None)

    is_training_multivariate = (
        True
        if (len(dku_config.target_columns_names) > 1) or (len(dku_config.timeseries_identifiers_names) > 0)
        else False
    )
    dku_config.add_param(name="is_training_multivariate", value=is_training_multivariate)

    if long_format and len(dku_config.timeseries_identifiers_names) == 0:
        raise PluginParamValidationError("Long format is activated but no time series identifiers have been provided")

    external_feature_activated = config.get("external_feature_activated", False)
    if external_feature_activated:
        dku_config.add_param(
            name="external_features_columns_names",
            value=sanitize_column_list(config.get("external_feature_columns", [])),
            checks=[
                {"type": "is_type", "op": list},
                {
                    "type": "is_subset",
                    "op": input_dataset_columns,
                    "err_msg": "Invalid external features selection: {}.".format(
                        config.get("external_feature_columns")
                    ),
                },
            ],
        )
    else:
        dku_config.add_param(name="external_features_columns_names", value=None)

    dku_config.add_param(name="frequency_unit", value=config.get("frequency_unit"))

    if dku_config.frequency_unit == "W":
        dku_config.add_param(name="frequency", value=f"W-{config.get('frequency_end_of_week', 'SUN')}")
    elif dku_config.frequency_unit == "H":
        dku_config.add_param(name="frequency", value=f"{config.get('frequency_step_hours', 1)}H")
    elif dku_config.frequency_unit == "min":
        dku_config.add_param(name="frequency", value=f"{config.get('frequency_step_minutes', 1)}min")
    else:
        dku_config.add_param(name="frequency", value=dku_config.frequency_unit)

    dku_config.add_param(
        name="prediction_length",
        label="Forecasting horizon",
        value=config.get("prediction_length"),
        checks=[
            {"type": "is_type", "op": int},
            {"type": "sup_eq", "op": 1},
        ],
        required=True,
    )

    dku_config.add_param(
        name="season_length",
        label="Seasonality",
        value=config.get(f"season_length_{dku_config.frequency_unit}", 1),
        checks=[
            {"type": "is_type", "op": int},
            {"type": "sup_eq", "op": 1},
        ],
        required=True,
    )

    dku_config.add_param(
        name="use_gpu",
        label="Use GPU",
        value=config.get("use_gpu", False),
    )

    if dku_config.use_gpu:
        dku_config.add_param(name="gpu_location", value=config.get("gpu_location", "local_gpu"))
        if dku_config.gpu_location == "local_gpu":
            gpu_devices = config.get("gpu_devices", [])
            dku_config.add_param(name="gpu_devices", value=parse_gpu_devices(gpu_devices))
        else:
            dku_config.add_param(name="gpu_devices", value=[GPU_CONFIGURATION.CONTAINER_GPU])
    else:
        dku_config.add_param(name="gpu_devices", value=None)

    dku_config.add_param(name="forecasting_style", value=config.get("forecasting_style", "auto"))
    dku_config.add_param(name="epoch", value=config.get("epoch", 10))
    dku_config.add_param(name="batch_size", value=config.get("batch_size", 32))

    dku_config.add_param(name="auto_num_batches_per_epoch", value=config.get("auto_num_batches_per_epoch", True))
    if dku_config.auto_num_batches_per_epoch:
        dku_config.add_param(name="num_batches_per_epoch", label="Number of batches per epoch", value=-1)
    else:
        print(f"recipe_config: {config}")
        dku_config.add_param(
            name="num_batches_per_epoch",
            label="Number of batches per epoch",
            value=config.get("num_batches_per_epoch"),
            required=True,
        )

    # Overwrite values in case of autoML mode selected
    automl_params_overwrite(dku_config)

    dku_config.add_param(name="sampling_method", value=config.get("sampling_method", "last_records"))

    dku_config.add_param(name="max_timeseries_length", value=None)

    if dku_config.sampling_method == "last_records":
        dku_config.add_param(
            name="max_timeseries_length",
            label="Number of records",
            value=config.get("number_records", 10000),
            checks=[
                {"type": "sup_eq", "op": 4},
            ],
        )

    dku_config.add_param(name="evaluation_strategy", value="split")
    dku_config.add_param(name="evaluation_only", value=False)


def add_predict_config(dku_config, config, file_manager):
    dku_config.add_param(
        name="partition_root", value=get_folder_partition_root(file_manager.model_folder, is_input=True)
    )

    check_only_one_read_partition(dku_config.partition_root, file_manager.model_folder)
    check_only_one_read_partition(dku_config.partition_root, file_manager.external_features_future_dataset)

    dku_config.add_param(
        name="manual_selection", value=True if config.get("model_selection_mode") == "manual" else False
    )

    dku_config.add_param(name="performance_metric", value=config.get("performance_metric"))

    dku_config.add_param(name="selected_session", value=config.get("manually_selected_session", "latest_session"))
    dku_config.add_param(name="selected_model_label", value=config.get("manually_selected_model_label"))

    dku_config.add_param(name="prediction_length", value=config.get("prediction_length", -1))
    dku_config.add_param(name="confidence_interval", value=config.get("confidence_interval", 95))
    dku_config.add_param(
        name="quantiles", value=convert_confidence_interval_to_quantiles(dku_config.confidence_interval)
    )
    dku_config.add_param(name="include_history", value=config.get("include_history", False))

    dku_config.add_param(name="sampling_method", value=config.get("sampling_method", "last_records"))
    dku_config.add_param(name="history_length_limit", value=None)

    if dku_config.sampling_method == "last_records":
        dku_config.add_param(
            name="history_length_limit",
            label="Number of historical records",
            value=config.get("number_records", 1000),
            checks=[
                {"type": "is_type", "op": int},
                {"type": "sup_eq", "op": 1},
            ],
        )


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


def get_column_names(dataset):
    dataset_columns = [column["name"] for column in dataset.read_schema()]
    return dataset_columns


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
    for model in ModelConfigRegistry().list_available_models():
        if is_activated(config, model, is_training_multivariate):
            model_presets = get_model_presets(config, model)
            if "prediction_length" in model_presets.get("kwargs", {}):
                raise ValueError(
                    "Keyword argument 'prediction_length' is not writable, please use the Forecasting horizon parameter"
                )
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
    forecasting_style = config.get("forecasting_style", "auto") + (
        "_multivariate" if is_training_multivariate else "_univariate"
    )
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


def automl_params_overwrite(dku_config):
    """Overwrite some training options based on the selected automl mode"""
    if dku_config.forecasting_style.startswith("auto"):
        dku_config.add_param(
            name="season_length",
            value=get_seasonality(dku_config.frequency, DEFAULT_SEASONALITIES),
        )
        dku_config.add_param(
            name="batch_size",
            value=128 if dku_config.use_gpu else 32,
        )

    if dku_config.forecasting_style == "auto":
        dku_config.add_param(name="epoch", value=10)
        dku_config.add_param(name="num_batches_per_epoch", value=50)
    elif dku_config.forecasting_style == "auto_performance":
        dku_config.add_param(name="context_length", value=dku_config.prediction_length)
        dku_config.add_param(name="epoch", value=30 if dku_config.is_training_multivariate else 10)
        dku_config.add_param(name="num_batches_per_epoch", value=-1)


def sanitize_column_list(input_column_list):
    """Remove empty elements (Nones, '') from input columns list"""
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
    """Keep the target list in same order as the training dataset, for consistency of forecasted columns order"""
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
        raise PluginParamValidationError(
            "GluonTS does not currently support multi-GPU training, please select only one GPU device"
        )
    else:  # one element list
        if gpu_devices[0] == GPU_CONFIGURATION.NO_GPU:
            raise PluginParamValidationError(
                "Local GPU device parameter is invalid, please check the CUDA/GPU installation on the DSS server"
            )
        else:
            return [int(gpu_device.split("_")[1]) for gpu_device in gpu_devices]
