from typing import NamedTuple, Optional
from functools import partial

# Third-party imports
import numpy as np
from mxnet.gluon import HybridBlock
from pydantic import ValidationError

# First-party imports
import gluonts
from gluonts.core import fqname_for
from gluonts.core.component import DType, from_hyperparameters, validated
from gluonts.core.exception import GluonTSHyperparametersError
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.model.predictor import Predictor
from gluonts.mx.trainer import Trainer
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.transform import Transformation
from gluonts.mx.batchify import batchify, as_in_context

from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


class TrainOutput(NamedTuple):
    predictor: Predictor
    trained_net: HybridBlock = None
    transformation: Transformation = None


def custom_train(
    estimator,
    training_data,
    update=False,
):
    if not update:
        predictor = estimator.train(training_data)
    else:
        logger.info("Update estimator by retraining")
        predictor = estimator.update(training_data)
    return TrainOutput(predictor=predictor)


def custom_train_nn(
    estimator,
    training_data,
    trained_net=None,
):
    transformation = estimator.create_transformation()

    training_data_loader = TrainDataLoader(
        dataset=training_data,
        transform=transformation,
        batch_size=estimator.trainer.batch_size,
        num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
        stack_fn=partial(
            batchify,
            ctx=estimator.trainer.ctx,
            dtype=estimator.dtype,
        ),
        decode_fn=partial(as_in_context, ctx=estimator.trainer.ctx),
    )

    # ensure that the training network is created within the same MXNet
    # context as the one that will be used during training
    if trained_net is None:
        with estimator.trainer.ctx:
            trained_net = estimator.create_training_network()
    else:
        estimator.trainer.epochs = max(1, int(estimator.trainer.epochs / 2))
        logger.info(f"Update estimator by retraining with {estimator.trainer.epochs} epochs")

    estimator.trainer(
        net=trained_net,
        input_names=get_hybrid_forward_input_names(trained_net),
        train_iter=training_data_loader,
        validation_iter=None,
    )

    with estimator.trainer.ctx:
        # ensure that the prediction network is created within the same MXNet
        # context as the one that was used during training
        return TrainOutput(
            predictor=estimator.create_predictor(transformation, trained_net),
            trained_net=trained_net,
            transformation=transformation,
        )