try:
    import mxnet as mx
except OSError as mxnet_or_cuda_error:  # error when importing mxnet
    raise Exception(f"Error when importing mxnet: {mxnet_or_cuda_error}")

from constants import GPU_CONFIGURATION


class GPUError(Exception):
    """Custom exception raised when the GPU selection failed"""

    pass


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
            raise GPUError(f"MXNet error: {num_gpus_error}, please check your server CUDA setup")

        if num_gpu == 0:
            if GPU_CONFIGURATION.CONTAINER_GPU in gpu_devices:
                raise GPUError("No GPU detected, please check that the container has GPUs")
            else:
                raise GPUError("No GPU detected, please check your server has GPUs")
        else:
            if GPU_CONFIGURATION.CONTAINER_GPU in gpu_devices:
                return mx.context.gpu(0)  # return first GPU of container
            else:
                return mx.context.gpu(gpu_devices[0])
