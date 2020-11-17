from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.trainer import Trainer
from gluonts.model.naive_2 import Naive2Predictor

ESTIMATOR = "estimator"
CAN_USE_EXTERNAL_FEATURES = "can_use_external_feature"
TRAINER = "trainer"
PREDICTOR = "predictor"

MODEL_DESCRIPTORS = {
    "default": {},
    "naive": {ESTIMATOR: None, PREDICTOR: Naive2Predictor, TRAINER: None},
    "simplefeedforward": {
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: SimpleFeedForwardEstimator,
        TRAINER: Trainer,
    },
    "deepar": {
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: DeepAREstimator,
        TRAINER: Trainer,
    },
    "transformer": {
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: TransformerEstimator,
        TRAINER: Trainer,
    },
    "nbeats": {ESTIMATOR: NBEATSEstimator, TRAINER: Trainer},
}


class ModelHandler:
    """
    Class to retrieve the estimator, trainer or descriptor of a GluonTS model

    Attributes:
        model_name (str)
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_descriptor = self._get_model_descriptor()

    def _get_model_descriptor(self):
        model_descriptor = MODEL_DESCRIPTORS.get(self.model_name)
        if model_descriptor is None:
            return MODEL_DESCRIPTORS.get("default")
        else:
            return model_descriptor

    def estimator(self, model_parameters, **kwargs):
        kwargs.update(model_parameters.get("kwargs", {}))
        estimator = self.model_descriptor.get(ESTIMATOR)
        return None if estimator is None else estimator(**kwargs)

    def trainer(self, **kwargs):
        trainer = self.model_descriptor.get(TRAINER)
        return None if trainer is None else trainer(**kwargs)

    def predictor(self, **kwargs):
        predictor = self.model_descriptor.get(PREDICTOR)
        return None if predictor is None else predictor(**kwargs)

    def can_use_external_feature(self):
        return self.model_descriptor.get(CAN_USE_EXTERNAL_FEATURES, False)
