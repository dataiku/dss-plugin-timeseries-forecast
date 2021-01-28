from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator

# from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.transformer import TransformerEstimator

# from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.trainer import Trainer

# from gluonts.model.trivial.mean import MeanPredictor
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.npts import NPTSPredictor
from gluonts_forecasts.custom_models.autoarima import AutoARIMAEstimator, AutoARIMAPredictor
from gluonts.mx.distribution import StudentTOutput, GaussianOutput, NegativeBinomialOutput
from gluonts_forecasts.utils import sanitize_model_parameters


ESTIMATOR = "estimator"
CAN_USE_EXTERNAL_FEATURES = "can_use_external_feature"
DEFAULT_KWARGS = "default_kwargs"
TRAINER = "trainer"
PREDICTOR = "predictor"
NEEDS_NUM_SAMPLES = "needs_num_samples"
LABEL = "label"
IS_NAIVE = "is_naive"


MODEL_DESCRIPTORS = {
    "trivial_identity": {
        LABEL: "TrivialIdentity",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: None,
        PREDICTOR: IdentityPredictor,
        TRAINER: None,
        NEEDS_NUM_SAMPLES: True,
        IS_NAIVE: True,
        DEFAULT_KWARGS: {"num_samples": 100},
    },
    "seasonal_naive": {
        LABEL: "SeasonalNaive",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: None,
        PREDICTOR: SeasonalNaivePredictor,
        TRAINER: None,
        IS_NAIVE: True,
    },
    "autoarima": {
        LABEL: "AutoARIMA",
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: AutoARIMAEstimator,
        PREDICTOR: AutoARIMAPredictor,
        TRAINER: None,
    },
    "npts": {
        LABEL: "NPTS",
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: None,
        PREDICTOR: NPTSPredictor,
        TRAINER: None,
        IS_NAIVE: True,
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
    "mqcnn": {
        LABEL: "MQ-CNN",
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: MQCNNEstimator,
        TRAINER: Trainer,
    },
}


# these parameter are classes but are set as strings in the UI
CLASS_PARAMETERS = {
    "distr_output": {"StudentTOutput()": StudentTOutput(), "GaussianOutput()": GaussianOutput(), "NegativeBinomialOutput()": NegativeBinomialOutput()}
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
            return {}
        else:
            return model_descriptor

    def estimator(self, model_parameters, **kwargs):
        default_kwargs = self.model_descriptor.get(DEFAULT_KWARGS, {})
        kwargs.update(default_kwargs)
        model_parameters_sanitized = sanitize_model_parameters(model_parameters.get("kwargs", {}), self.model_name)
        kwargs.update(model_parameters_sanitized)
        estimator = self.model_descriptor.get(ESTIMATOR)
        kwargs = self._convert_parameters_to_class(kwargs)
        try:
            ret = None if estimator is None else estimator(**kwargs)
        except Exception as err:
            raise ModelParameterError(f"Issue with parameters '{kwargs}' of model '{self.model_name}'. Full error: {err}")
        return ret

    def trainer(self, **kwargs):
        trainer = self.model_descriptor.get(TRAINER)
        try:
            ret = None if trainer is None else trainer(**kwargs)
        except Exception as err:
            raise ModelParameterError(f"Issue with parameters '{kwargs}' of model trainer for '{self.model_name}'. Full error: {err}")
        return ret

    def predictor(self, **kwargs):
        predictor = self.model_descriptor.get(PREDICTOR)
        try:
            ret = None if predictor is None else predictor(**kwargs)
        except Exception as err:
            raise ModelParameterError(f"Issue with parameters '{kwargs}' of model predictor for {self.model_name}. Full error: {err}")
        return ret

    def can_use_external_feature(self):
        return self.model_descriptor.get(CAN_USE_EXTERNAL_FEATURES, False)

    def needs_num_samples(self):
        return self.model_descriptor.get(NEEDS_NUM_SAMPLES, False)

    def get_label(self):
        return self.model_descriptor.get(LABEL, "")

    def _convert_parameters_to_class(self, parameters):
        """Evaluate string parameters that are classes so that they become instances of their class """
        parameters_converted = parameters.copy()
        for class_parameter, class_parameter_values in CLASS_PARAMETERS.items():
            if class_parameter in parameters_converted:
                if parameters_converted[class_parameter] not in class_parameter_values:
                    raise ModelParameterError(
                        f"""
                        '{parameters_converted[class_parameter]}' is not valid for parameter '{class_parameter}'.
                        Supported values are {list(class_parameter_values.keys())}. 
                    """
                    )
                else:
                    parameters_converted[class_parameter] = class_parameter_values[parameters_converted[class_parameter]]
        return parameters_converted


def list_available_models():
    """List available models names found in the recipe.json (keys of MODEL_DESCRIPTORS).

    Returns:
        dict_keys of model names.
    """
    available_models = MODEL_DESCRIPTORS.copy()
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
