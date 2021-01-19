# -*- coding: utf-8 -*-
import re
import dataiku
import mxnet as mx
from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


def do(payload, config, plugin_config, inputs):
    """ Retrieve a list of available GPU devices. """
    choices = []

    if payload.get("parameterName") == "gpu_devices":
        try:
            num_gpu = mx.context.num_gpus()
        except mx.base.MXNetError as e:
            logger.error(f"Error when querying number of GPU: {e}")
            choices = [{"label": "Fail querying GPUs, please check your CUDA installation", "value": "no_gpu_available"}]
        else:
            if num_gpu > 0:
                choices = [{"label": f"GPU #{n}", "value": n} for n in range(num_gpu)]
            else:
                choices = [{"label": "No GPU available, please check your CUDA installation", "value": "no_gpu_available"}]

    return {"choices": choices}
