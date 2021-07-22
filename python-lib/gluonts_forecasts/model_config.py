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


class ModelConfig:
    """
    Class to retrieve the estimator, trainer or descriptor of a GluonTS model.
    By default, models have no special behaviors (such as batch size, external features, seasonality).
    """

    def __init__(self, model_name, label, estimator_class=None, predictor_class=None, trainer_class=None):
        self.model_name = model_name
        self.label = label
        self.estimator_class = estimator_class
        self.predictor_class = predictor_class
        self.trainer_class = trainer_class

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


class TrivialIdentity(ModelConfig):
    model_name = "trivial_identity"

    def __init__(self):
        super().__init__(model_name=self.model_name, label="TrivialIdentity", predictor_class=IdentityPredictor)

    def needs_num_samples(self):
        return True

    def estimator(self, model_parameters, **kwargs):
        default_kwargs = {"num_samples": 100}
        kwargs.update(default_kwargs)
        return super().estimator(model_parameters, **kwargs)


class SeasonalNaive(ModelConfig):
    model_name = "seasonal_naive"

    def __init__(self):
        super().__init__(model_name=self.model_name, label="SeasonalNaive", predictor_class=SeasonalNaivePredictor)

    def can_use_seasonality(self):
        return True


class AutoARIMA(ModelConfig):
    model_name = "autoarima"

    def __init__(self):
        super().__init__(
            model_name=self.model_name,
            label="AutoARIMA",
            estimator_class=AutoARIMAEstimator,
            predictor_class=AutoARIMAPredictor,
        )

    def can_use_seasonality(self):
        return True

    def can_use_external_feature(self):
        return True


class SeasonalTrend(ModelConfig):
    model_name = "seasonal_trend"

    def __init__(self):
        super().__init__(
            model_name=self.model_name,
            label="SeasonalTrend",
            estimator_class=SeasonalTrendEstimator,
            predictor_class=SeasonalTrendPredictor,
        )

    def can_use_seasonality(self):
        return True


class NPTS(ModelConfig):
    model_name = "npts"

    def __init__(self):
        super().__init__(model_name=self.model_name, label="NPTS", predictor_class=NPTSPredictor)


class FeedForward(ModelConfig):
    model_name = "simplefeedforward"

    def __init__(self):
        super().__init__(
            model_name=self.model_name,
            label="FeedForward",
            estimator_class=SimpleFeedForwardEstimator,
            trainer_class=Trainer,
        )

    def can_use_batch_size(self):
        return True


class DeepAR(ModelConfig):
    model_name = "deepar"

    def __init__(self):
        super().__init__(
            model_name=self.model_name, label="DeepAR", estimator_class=DeepAREstimator, trainer_class=Trainer
        )

    def can_use_external_feature(self):
        return True

    def can_use_batch_size(self):
        return True


class Transformer(ModelConfig):
    model_name = "transformer"

    def __init__(self):
        super().__init__(
            model_name=self.model_name, label="Transformer", estimator_class=TransformerEstimator, trainer_class=Trainer
        )

    def can_use_external_feature(self):
        return True

    def can_use_batch_size(self):
        return True


class MQCNN(ModelConfig):
    model_name = "mqcnn"

    def __init__(self):
        super().__init__(
            model_name=self.model_name, label="MQ-CNN", estimator_class=MQCNNEstimator, trainer_class=Trainer
        )

    def can_use_external_feature(self):
        return True
