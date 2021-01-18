# -*- coding: utf-8 -*-
import re
import dataiku
from gluonts.mx.context import num_gpus
import mxnet as mx


def do(payload, config, plugin_config, inputs):
    """ Retrieve a list of available GPU devices. """
    choices = []

    if payload.get("parameterName") == "gpu_devices":
        num_gpu = num_gpus(refresh=True)
        if num_gpu > 0:
            choices += [{"label": f"GPU #{n}", "value": n} for n in range(num_gpu)]
        else:
            choices += [{"label": "No GPU available, please check your CUDA installation", "value": "no_gpu_available"}]

    return {"choices": choices}
