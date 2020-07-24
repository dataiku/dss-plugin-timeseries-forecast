# -*- coding: utf-8 -*-
# TODO


def do(payload, config, plugin_config, inputs):
    choices = []
    if payload.get("parameterName") == "manually_selected_session":
        choices = [{"label": "1995", "value": "1995"}]
    elif payload.get("parameterName") == "manually_selected_model_type":
        choices = [{"label": "JohnWickEstimator", "value": "JohnWickEstimator"}]
    else:
        choices += [{"label": "foo", "value": "bar"}]
    return {"choices": choices}
