# -*- coding: utf-8 -*-
from dku_constants import GPU_CONFIGURATION
from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


def do(payload, config, plugin_config, inputs):
    """ Retrieve a list of available GPU devices. """
    choices = []

    if payload.get("parameterName") == "cpu_devices":
        try:
            import mxnet as mx
        except OSError as mxnet_or_cuda_error:  # error when importing mxnet
            logger.error(f"Error when importing mxnet: {mxnet_or_cuda_error}")
            choices += [
                {
                    "label": f"No GPU detected on DSS server, please check it has CUDA {GPU_CONFIGURATION.CUDA_VERSION} installed",
                    "value": GPU_CONFIGURATION.NO_GPU,
                }
            ]
        else:
            try:
                num_gpu = mx.context.num_gpus()
            except mx.base.MXNetError as num_gpus_error:  # error on num_gpus()
                logger.error(f"Cuda error: {num_gpus_error}")
                choices += [
                    {
                        "label": f"No GPU detected on DSS server, please check it has CUDA {GPU_CONFIGURATION.CUDA_VERSION} installed",
                        "value": GPU_CONFIGURATION.NO_GPU,
                    }
                ]
            else:
                if num_gpu > 0:
                    choices += [{"label": f"GPU #{n}", "value": f"gpu_{n}"} for n in range(num_gpu)]
                else:
                    choices += [
                        {
                            "label": f"No GPU detected on DSS server, please check that CUDA can access the GPU(s) with the NVIDA driver",
                            "value": GPU_CONFIGURATION.NO_GPU,
                        }
                    ]

    return {"choices": choices}
