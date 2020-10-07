import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


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

    params['external_features_future'] = None
    external_features_future_dataset_names = get_input_names_for_role('external_features_future_future_dataset')
    if len(external_features_future_dataset_names) > 0:
        params['external_features_future'] = dataiku.Dataset(external_features_future_dataset_names[0])

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise ValueError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])

    params['manual_selection'] = True if recipe_config.get("model_selection_mode") == "manual" else False

    params['performance_metric'] = recipe_config.get("performance_metric")
    params['selected_session'] = recipe_config.get("manually_selected_session")
    params['selected_model_type'] = recipe_config.get("manually_selected_model_type")

    params['forecasting_horizon'] = recipe_config.get("forecasting_horizon")
    params['confidence_interval_1'] = recipe_config.get("confidence_interval_1")/100
    params['confidence_interval_2'] = recipe_config.get("confidence_interval_2")/100

    return params


def load_training_config(recipe_config):
    params = {}
    params['input_dataset_name'] = get_input_names_for_role('input_dataset')[0]
    model_folder_name = get_output_names_for_role('model_folder')[0]
    params['model_folder'] = dataiku.Folder(model_folder_name)
    params['evaluation_dataset_name'] = get_output_names_for_role('evaluation_dataset')[0]

    evaluation_forecasts_names = get_output_names_for_role("evaluation_forecasts")
    if len(evaluation_forecasts_names) > 0:
        params["evaluation_forecasts"] = dataiku.Dataset(evaluation_forecasts_names[0])

    params['target_columns_names'] = recipe_config.get("target_columns")
    params['time_column_name'] = recipe_config.get("time_column")
    params['external_feature_columns'] = recipe_config.get('external_feature_columns', [])
    params['deepar_model_activated'] = recipe_config.get('deepar_model_activated', False)
    params['time_granularity_unit'] = recipe_config.get('time_granularity_unit')
    params['time_granularity_step'] = recipe_config.get('time_granularity_step', 1)
    params['frequency'] = "{}{}".format(params['time_granularity_step'], params['time_granularity_unit'])

    params['prediction_length'] = 10
    params['epoch'] = 10

    params['evaluation_dataset'] = dataiku.Dataset(params['evaluation_dataset_name'])
    params['evaluation_strategy'] = recipe_config.get("evaluation_strategy", "split")
    params['forecasting_horizon'] = recipe_config.get("forecasting_horizon", 1)
    return params
