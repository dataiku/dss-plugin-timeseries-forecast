# -*- coding: utf-8 -*-
import re
import dataiku
from dku_constants import TIMESTAMP_REGEX_PATTERN
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from dku_io_utils.model_selection import ModelSelection


def do(payload, config, plugin_config, inputs):
    """Retrieve a list of past training session timestamps and the label of all the trained models."""
    choices = []

    input_folder_name = [input["fullName"] for input in inputs if input["type"] == "MANAGED_FOLDER"][0]

    input_folder = dataiku.Folder(input_folder_name)

    sessions = []
    for child in input_folder.get_path_details(path="/")["children"]:
        if re.match(TIMESTAMP_REGEX_PATTERN, child["name"]):
            sessions += [child["name"]]

    if payload.get("parameterName") == "manually_selected_session":
        choices = [{"label": "Latest available", "value": "latest_session"}]
        if len(sessions) > 0:  # not partitioned folder
            for i, session in enumerate(sorted(sessions, reverse=True)):
                choices += [{"label": session, "value": session}]

    elif payload.get("parameterName") == "manually_selected_model_label":
        model_labels = ModelSelection.find_all_models_labels_from_folder(input_folder)
        choices = [{"label": "All models", "value": "all_models"}]
        for model_label in model_labels:
            choices += [{"label": model_label, "value": model_label}]

    elif payload.get("parameterName") == "model_selection_mode":
        choices = [{"label": "Automatic", "value": "auto"}, {"label": "Manual", "value": "manual"}]

    return {"choices": choices}
