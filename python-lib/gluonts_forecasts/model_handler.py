from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.trainer import Trainer
from gluonts.model.trivial.mean import MeanEstimator, MeanPredictor
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor


ESTIMATOR = "estimator"
CAN_USE_EXTERNAL_FEATURES = "can_use_external_feature"
CAN_USE_CONTEXT_LENGTH = "can_use_context_length"
TRAINER = "trainer"
PREDICTOR = "predictor"
NEEDS_NUM_SAMPLES = "needs_num_samples"
LABEL = "label"


MODEL_DESCRIPTORS = {
    "default": {},
    "trivial_identity": {
        LABEL: "TrivialIdentity",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: None,
        PREDICTOR: IdentityPredictor,
        TRAINER: None,
        CAN_USE_CONTEXT_LENGTH: False,
        NEEDS_NUM_SAMPLES: True,
    },
    "trivial_mean": {
        LABEL: "TrivialMean",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: MeanEstimator,
        PREDICTOR: MeanPredictor,
        TRAINER: None,
        CAN_USE_CONTEXT_LENGTH: False
    },
    "seasonal_naive": {
        LABEL: "SeasonalNaive",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: None,
        PREDICTOR: SeasonalNaivePredictor,
        TRAINER: None,
        CAN_USE_CONTEXT_LENGTH: False
    },
    "simplefeedforward": {
        LABEL: "FeedForward",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: SimpleFeedForwardEstimator,
        TRAINER: Trainer,
    },
    "deepar": {
        LABEL: "DeepAR",
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: DeepAREstimator,
        TRAINER: Trainer,
    },
    "transformer": {
        LABEL: "Transformer",
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: TransformerEstimator,
        TRAINER: Trainer,
    },
    "nbeats": {
        LABEL: "NBEATS",
        ESTIMATOR: NBEATSEstimator,
        TRAINER: Trainer
    },
    "mqcnn": {
        LABEL: "MQ-CNN",
        ESTIMATOR: MQCNNEstimator,
        TRAINER: Trainer
    },
    "tft": {
        LABEL: "TemporalFusionTransformer",
        ESTIMATOR: TemporalFusionTransformerEstimator,
        TRAINER: Trainer
    },
}


class ModelParameterError(ValueError):
    """Custom exception raised when the GluonTS model parameters chosen by the user are invalid"""
    pass


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
        try:
            ret = None if estimator is None else estimator(**kwargs)
        except Exception as err:
            raise ModelParameterError("Issue with the parameters ({}) for model '{}' estimator: {}".format(kwargs, self.model_name, err))
        return ret

    def trainer(self, **kwargs):
        trainer = self.model_descriptor.get(TRAINER)
        try:
            ret = None if trainer is None else trainer(**kwargs)
        except Exception as err:
            raise ModelParameterError("Issue with the parameters ({}) for model '{}' trainer: {}".format(kwargs, self.model_name, err))
        return ret

    def predictor(self, **kwargs):
        predictor = self.model_descriptor.get(PREDICTOR)
        try:
            ret = None if predictor is None else predictor(**kwargs)
        except Exception as err:
            raise ModelParameterError("Issue with the parameters ({}) for model '{}' predictor: {}".format(kwargs, self.model_name, err))
        return ret

    def can_use_external_feature(self):
        return self.model_descriptor.get(CAN_USE_EXTERNAL_FEATURES, False)

    def can_use_context_length(self):
        return self.model_descriptor.get(CAN_USE_CONTEXT_LENGTH, True)

    def needs_num_samples(self):
        return self.model_descriptor.get(NEEDS_NUM_SAMPLES, False)

    def get_label(self):
        return self.model_descriptor.get(LABEL, "")


def list_available_models():
    """List available models names found in the recipe.json (keys of MODEL_DESCRIPTORS).

    Returns:
        dict_keys of model names.
    """
    available_models = MODEL_DESCRIPTORS.copy()
    available_models.pop('default')
    return available_models.keys()


def list_available_models_labels():
    """List available models labels found in the UI.

    Returns:
        List of model names.
    """
    available_models = MODEL_DESCRIPTORS.copy()
    available_models_labels = []
    for model in available_models:
        label = available_models[model].get(LABEL)
        if label is not None:
            available_models_labels.append(label)
    return available_models_labels


def get_model_label(model_name):
    model_descriptor = MODEL_DESCRIPTORS.get(model_name)
    if model_descriptor is None:
        return None
    else:
        return model_descriptor.get(LABEL, "")
