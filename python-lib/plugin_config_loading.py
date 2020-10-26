import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
import plugin_io_utils as utils


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""
    pass


def load_predict_config():
    """Utility function to load, resolve and validate all plugin config into a clean `params` dictionary

    Returns:
        Dictionary of parameter names (key) and values
    """
    params = {}
    recipe_config = get_recipe_config()

    # input folder
    model_folder = dataiku.Folder(get_input_names_for_role('model_folder')[0])
    params['model_folder'] = model_folder

    params['external_features_future_dataset'] = None
    external_features_future_dataset_names = get_input_names_for_role('external_features_future_dataset')
    if len(external_features_future_dataset_names) > 0:
        params['external_features_future_dataset'] = dataiku.Dataset(external_features_future_dataset_names[0])

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])

    params['manual_selection'] = True if recipe_config.get("model_selection_mode") == "manual" else False

    params['performance_metric'] = recipe_config.get("performance_metric")
    params['selected_session'] = recipe_config.get("manually_selected_session")
    params['selected_model_type'] = recipe_config.get("manually_selected_model_type")

    params['forecasting_horizon'] = recipe_config.get("forecasting_horizon")
    params['quantiles'] = recipe_config.get("quantiles")
    if any(x < 0 or x > 1 for x in params['quantiles']):
        raise PluginParamValidationError("Quantiles must be between 0 and 1.")
    params['quantiles'].sort()

    params['include_history'] = recipe_config.get("include_history")

    return params


def load_training_config(recipe_config):
    params = {}

    input_dataset_name = get_input_names_for_role('input_dataset')[0]
    params['training_dataset'] = dataiku.Dataset(input_dataset_name)
    training_dataset_columns = [p["name"] for p in params["training_dataset"].read_schema()]

    model_folder_name = get_output_names_for_role('model_folder')[0]
    params['model_folder'] = dataiku.Folder(model_folder_name)

    evaluation_dataset_name = get_output_names_for_role('evaluation_dataset')[0]
    params['evaluation_dataset'] = dataiku.Dataset(evaluation_dataset_name)

    params['make_forecasts'] = False
    evaluation_forecasts_names = get_output_names_for_role("evaluation_forecasts")
    if len(evaluation_forecasts_names) > 0:
        params["evaluation_forecasts"] = dataiku.Dataset(evaluation_forecasts_names[0])
        params['make_forecasts'] = True

    params['time_column_name'] = recipe_config.get("time_column")
    if params['time_column_name'] not in training_dataset_columns:
        raise PluginParamValidationError("Invalid time column selection")

    params['target_columns_names'] = recipe_config.get("target_columns")
    if len(params['target_columns_names']) == 0 or not all(column in training_dataset_columns for column in params['target_columns_names']):
        raise PluginParamValidationError("Invalid target column(s) selection")

    params['external_feature_columns'] = recipe_config.get('external_feature_columns', [])
    if not all(column in training_dataset_columns for column in params['external_feature_columns']):
        raise PluginParamValidationError("Invalid external features column(s) selection")

    params['deepar_model_activated'] = recipe_config.get('deepar_model_activated', False)
    params['time_granularity_unit'] = recipe_config.get('time_granularity_unit')
    params['time_granularity_step'] = recipe_config.get('time_granularity_step', 1)
    params['frequency'] = "{}{}".format(params['time_granularity_step'], params['time_granularity_unit'])

    # order of cols is important (for the predict recipe)
    params['columns_to_keep'] = [params['time_column_name']] + params['target_columns_names'] + params['external_feature_columns']

    params['prediction_length'] = recipe_config.get('forecasting_horizon', 30)
    params['epoch'] = recipe_config.get('epoch', 1)

    params['evaluation_strategy'] = recipe_config.get("evaluation_strategy", "split")

    utils.assert_time_column_is_date(params['training_dataset'], params['time_column_name'])

    return params
