from constants import GPU_CONFIGURATION


class GPUError(Exception):
    """Custom exception raised when the GPU selection failed"""

    pass


try:
    import mxnet as mx
except OSError as mxnet_or_cuda_error:  # error when importing mxnet
    raise GPUError(
        "Error when importing mxnet, please check that "
        + f"you have CUDA {GPU_CONFIGURATION.CUDA_VERSION} installed. "
        + f"Detailed error: {mxnet_or_cuda_error}"
    )


def set_mxnet_context(gpu_devices):
    """Return the right MXNet context from the selected GPU configuration.

    Args:
        gpu_devices (list): List of gpu device numbers or 'container_gpu'. Default to None which means no gpu.

    Raises:
        GPUError:
            If mx.context.num_gpus() fails on a Cuda error.
            If no GPUs are detected on the DSS server or in the container.

    Returns:
        A mxnet.context.Context to use for Deep Learning models training.
    """

    if gpu_devices is None:
        return mx.context.cpu()
    else:
        try:
            num_gpu = mx.context.num_gpus()
        except mx.base.MXNetError as num_gpus_error:  # error on num_gpus()
            raise GPUError(
                "Error when detecting GPUs, please check that "
                + f"you have CUDA {GPU_CONFIGURATION.CUDA_VERSION} installed. "
                + f"Detailed error: {num_gpus_error}"
            )

        if num_gpu == 0:
            if GPU_CONFIGURATION.CONTAINER_GPU in gpu_devices:
                raise GPUError("No GPU detected on DSS server, please check that CUDA can access the GPU(s) with the NVIDA driver")
            else:
                raise GPUError("No GPU detected on container, please check that CUDA can access the GPU(s) with the NVIDA driver")
        else:
            if GPU_CONFIGURATION.CONTAINER_GPU in gpu_devices:
                return mx.context.gpu(0)  # return first GPU of container
            else:
                return mx.context.gpu(gpu_devices[0])
