from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


def load_predict_config() -> Dict:
    """Utility function to load, resolve and validate all plugin config into a clean `params` dictionary

    Returns:
        Dictionary of parameter names (key) and values
    """
    params = {}
    recipe_config = get_recipe_config()

    # input folder
    input_folder = dataiku.Folder(get_input_names_for_role('input_folder')[0])
    params['input_folder'] = input_folder

    params['manual_selection'] = True if recipe_config.get("model_selection_mode") == "manual" else False

    params['performance_metric'] = recipe_config.get("performance_metric")
    params['selected_session'] = recipe_config.get("manually_selected_session")
    params['selected_model_type'] = recipe_config.get("manually_selected_model_type")

    params['forecasting_horizon'] = recipe_config.get("forecasting_horizon")
    params['confidence_interval_1'] = recipe_config.get("confidence_interval_1")
    params['confidence_interval_2'] = recipe_config.get("confidence_interval_2")

    return params
