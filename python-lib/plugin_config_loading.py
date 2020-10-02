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
