# -*- coding: utf-8 -*-
import re
import dataiku
from constants import AVAILABLE_MODELS


def do(payload, config, plugin_config, inputs):
    """
    retrieve a list of models from the summary.csv file of the S3 Input Folder
    """
    choices = []

    input_folder_name = [input['fullName'] for input in inputs if input['type'] == "MANAGED_FOLDER"][0]

    input_folder = dataiku.Folder(input_folder_name)

    sessions = []
    for child in input_folder.get_path_details(path='/')['children']:
        if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z', child['name']):
            sessions += [child['name']]

    if payload.get('parameterName') == 'manually_selected_session':
        for session in sessions:
            choices += [{'label': session, 'value': session}]

    elif payload.get('parameterName') == 'manually_selected_model_type':
        model_types = set()
        for session in sessions:
            for child in input_folder.get_path_details(path='/{}'.format(session))['children']:
                if child['directory'] and child['name'] in AVAILABLE_MODELS:
                    model_types.add(child['name'])

        for model in model_types:
            choices += [{'label': model, 'value': model}]

    elif payload.get('parameterName') == 'model_selection_mode':
        choices = [{'label': 'Automatic', 'value': 'auto'}]
        if len(sessions) > 0:
            choices += [{'label': 'Manual', 'value': 'manual'}]

    return {"choices": choices}
