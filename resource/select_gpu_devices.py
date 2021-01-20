# -*- coding: utf-8 -*-
import re
import dataiku
from constants import GPU_CONFIGURATION
from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


def do(payload, config, plugin_config, inputs):
    """ Retrieve a list of available GPU devices. """
    choices = []

    if payload.get("parameterName") == "gpu_devices":
        try:
            import mxnet as mx
        except OSError as cuda_error:  # error when importing mxnet
            logger.error(f"Error when importing mxnet: {cuda_error}")
            choices += [{"label": "No GPU detected on DSS server, please check your server CUDA installation", "value": GPU_CONFIGURATION.NO_GPU}]
        else:
            try:
                num_gpu = mx.context.num_gpus()
            except mx.base.MXNetError as num_gpus_error:  # error on num_gpus()
                logger.error(f"Cuda error: {num_gpus_error}")
                choices += [{"label": "No GPU detected on DSS server, please check your server CUDA installation", "value": GPU_CONFIGURATION.NO_GPU}]
            else:
                if num_gpu > 0:
                    choices += [{"label": f"GPU #{n}", "value": f"gpu_{n}"} for n in range(num_gpu)]
                else:
                    choices += [{"label": "No GPU detected on DSS server, please check that your server has GPUs", "value": GPU_CONFIGURATION.NO_GPU}]

    return {"choices": choices}
