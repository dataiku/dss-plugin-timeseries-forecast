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
from gluonts_forecasts.custom_models.seasonal_trend import SeasonalTrendEstimator, SeasonalTrendPredictor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from gluonts.mx.distribution import StudentTOutput, GaussianOutput, NegativeBinomialOutput
from gluonts_forecasts.utils import sanitize_model_parameters


# these parameter are classes but are set as strings in the UI
CLASS_PARAMETERS = {
    "distr_output": {
        "StudentTOutput()": StudentTOutput(),
        "GaussianOutput()": GaussianOutput(),
        "NegativeBinomialOutput()": NegativeBinomialOutput(),
    },
    "model": {"ARIMA": ARIMA, "ETSModel": ETSModel},
}


class ModelParameterError(ValueError):
    """Custom exception raised when the GluonTS model parameters chosen by the user are invalid"""

    pass


class ModelHandler:
    """
    Class to retrieve the estimator, trainer or descriptor of a GluonTS model
    """

    def __init__(self):
        self.estimator_class = None
        self.predictor_class = None
        self.trainer_class = None

    def estimator(self, model_parameters, **kwargs):
        model_parameters_sanitized = sanitize_model_parameters(model_parameters.get("kwargs", {}), self.model_name)
        kwargs.update(model_parameters_sanitized)
        kwargs = self._convert_parameters_to_class(kwargs)
        try:
            ret = None if self.estimator_class is None else self.estimator_class(**kwargs)
        except Exception as err:
            raise ModelParameterError(f"Issue with parameters '{kwargs}' of model '{self.label}'. Full error: {err}")
        return ret

    def trainer(self, **kwargs):
        try:
            ret = None if self.trainer_class is None else self.trainer_class(**kwargs)
        except Exception as err:
            raise ModelParameterError(
                f"Issue with parameters '{kwargs}' of model trainer for '{self.label}'. Full error: {err}"
            )
        return ret

    def predictor(self, **kwargs):
        try:
            ret = None if self.predictor_class is None else self.predictor_class(**kwargs)
        except Exception as err:
            raise ModelParameterError(
                f"Issue with parameters '{kwargs}' of model predictor for {self.label}. Full error: {err}"
            )
        return ret

    def can_use_batch_size(self):
        return False

    def can_use_external_feature(self):
        return False

    def can_use_seasonality(self):
        return False

    def needs_num_samples(self):
        return False

    def get_label(self):
        return self.label

    def _convert_parameters_to_class(self, parameters):
        """Evaluate string parameters that are classes so that they become instances of their class"""
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
                    parameters_converted[class_parameter] = class_parameter_values[
                        parameters_converted[class_parameter]
                    ]
        return parameters_converted


class TrivialIdentity(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "trivial_identity"
        self.label = "TrivialIdentity"
        self.predictor_class = IdentityPredictor

    def needs_num_samples(self):
        return True

    def estimator(self, model_parameters, **kwargs):
        default_kwargs = {"num_samples": 100}
        kwargs.update(default_kwargs)
        return super().estimator(model_parameters, **kwargs)


class SeasonalNaive(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "seasonal_naive"
        self.label = "SeasonalNaive"
        self.predictor_class = SeasonalNaivePredictor

    def can_use_seasonality(self):
        return True


class AutoARIMA(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "autoarima"
        self.label = "AutoARIMA"
        self.estimator_class = AutoARIMAEstimator
        self.predictor_class = AutoARIMAPredictor

    def can_use_seasonality(self):
        return True

    def can_use_external_feature(self):
        return True


class SeasonalTrend(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "seasonal_trend"
        self.label = "SeasonalTrend"
        self.estimator_class = SeasonalTrendEstimator
        self.predictor_class = SeasonalTrendPredictor

    def can_use_seasonality(self):
        return True


class NPTS(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "npts"
        self.label = "NPTS"
        self.predictor_class = NPTSPredictor


class FeedForward(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "simplefeedforward"
        self.label = "FeedForward"
        self.estimator_class = SimpleFeedForwardEstimator
        self.trainer_class = Trainer

    def can_use_batch_size(self):
        return True


class DeepAR(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "deepar"
        self.label = "DeepAR"
        self.estimator_class = DeepAREstimator
        self.trainer_class = Trainer

    def can_use_external_feature(self):
        return True

    def can_use_batch_size(self):
        return True


class Transformer(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "transformer"
        self.label = "Transformer"
        self.estimator_class = TransformerEstimator
        self.trainer_class = Trainer

    def can_use_external_feature(self):
        return True

    def can_use_batch_size(self):
        return True


class MQCNN(ModelHandler):
    def __init__(self):
        super().__init__()
        self.model_name = "mqcnn"
        self.label = "MQ-CNN"
        self.estimator_class = MQCNNEstimator
        self.trainer_class = Trainer

    def can_use_external_feature(self):
        return True


ModelHandlerRegistry = {
    "trivial_identity": TrivialIdentity(),
    "seasonal_naive": SeasonalNaive(),
    "autoarima": AutoARIMA(),
    "seasonal_trend": SeasonalTrend(),
    "npts": NPTS(),
    "simplefeedforward": FeedForward(),
    "deepar": DeepAR(),
    "transformer": Transformer(),
    "mqcnn": MQCNN(),
}


def list_available_models():
    """List available models names found in the recipe.json (keys of MODEL_DESCRIPTORS).

    Returns:
        dict_keys of model names.
    """
    available_models = ModelHandlerRegistry.copy()
    return available_models.keys()


def list_available_models_labels():
    """List available models labels found in the UI.

    Returns:
        List of model names.
    """
    available_models = ModelHandlerRegistry.copy()
    available_models_labels = []
    for model in available_models:
        label = available_models[model].get_label()
        if label is not None:
            available_models_labels.append(label)
    return available_models_labels


def get_model_label(model_name):
    model_handler = ModelHandlerRegistry.get(model_name)
    if model_handler is None:
        return None
    else:
        return model_handler.get_label()


def get_model_name_from_label(model_label):
    available_models = ModelHandlerRegistry.copy()
    return next(
        (model_name for model_name in available_models if available_models[model_name].get_label() == model_label), None
    )
