# -*- coding: utf-8 -*-
import re
import dataiku
from constants import TIMESTAMP_REGEX_PATTERN
from gluonts_forecasts.model_handler import list_available_models_labels


def do(payload, config, plugin_config, inputs):
    """ Retrieve a list of past training session timestamps and the label of all the trained models. """
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
        all_paths = input_folder.list_paths_in_partition()
        for model_label in list_available_models_labels():
            for path in all_paths:
                if bool(re.search(f"({model_label})(/model.pk.gz)", path)):
                    choices += [{"label": model_label, "value": model_label}]
                    break

    elif payload.get("parameterName") == "model_selection_mode":
        choices = [{"label": "Automatic", "value": "auto"}, {"label": "Manual", "value": "manual"}]

    return {"choices": choices}
