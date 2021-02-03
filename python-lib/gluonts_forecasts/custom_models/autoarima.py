from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import frequency_add
from gluonts.core.component import validated
from gluonts_forecasts.custom_models.utils import cast_kwargs
from constants import TIMESERIES_KEYS
from pmdarima.arima.utils import nsdiffs
import pmdarima as pm
import numpy as np
from safe_logger import SafeLogger
from tqdm import tqdm
from threadpoolctl import threadpool_limits


logger = SafeLogger("Forecast plugin - AutoARIMA")


class AutoARIMAPredictor(RepresentablePredictor):
    """
    An abstract predictor that can be subclassed by models that are not based
    on Gluon. Subclasses should have @validated() constructors.
    (De)serialization and value equality are all implemented on top of the
    @validated() logic.

    Parameters
    ----------
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    """

    @validated()
    def __init__(self, prediction_length, freq, trained_models, lead_time=0):
        super().__init__(freq=freq, lead_time=lead_time, prediction_length=prediction_length)
        self.trained_models = trained_models

    def predict(self, dataset, **kwargs):
        """

        Args:
            dataset (gluonts.dataset.common.Dataset): Dataset after wich to predict forecasts.

        Yields:
            SampleForecast of predictions.
        """
        logger.info("Predicting time series ...")
        for i, item in tqdm(enumerate(dataset)):
            yield self.predict_item(item, self.trained_models[i])

    def predict_item(self, item, trained_model):
        """Compute quantiles using the confidence intervals of autoarima.

        Args:
            item (DataEntry): One timeseries.
            trained_model (list): List of trained autoarima models.

        Returns:
            SampleForecast of quantiles.
        """
        start_date = frequency_add(item[TIMESERIES_KEYS.START], len(item[TIMESERIES_KEYS.TARGET]))

        prediction_external_features = self._set_prediction_external_features(item)

        samples = []
        for alpha in np.arange(0.02, 1.01, 0.02):
            confidence_intervals = trained_model.predict(n_periods=self.prediction_length, X=prediction_external_features, return_conf_int=True, alpha=alpha)[1]
            samples += [confidence_intervals[:, 0], confidence_intervals[:, 1]]

        return SampleForecast(samples=np.stack(samples), start_date=start_date, freq=self.freq)

    def _set_prediction_external_features(self, item):
        prediction_external_features = None
        if TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES in item:
            prediction_external_features = item[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL][:, -self.prediction_length :].T
        return prediction_external_features


class AutoARIMAEstimator(Estimator):
    @validated()
    def __init__(self, prediction_length, freq, season_length=None, use_feat_dynamic_real=False, **kwargs):
        super().__init__()
        self.prediction_length = prediction_length
        self.freq = freq
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.kwargs = cast_kwargs(kwargs)
        self.thread_limit = 1
        if "m" in kwargs:
            raise ValueError("Keyword argument 'm' is not writable for AutoARIMA, please use the Seasonality parameter")
        self.season_length = season_length if season_length is not None else 1

    def train(self, training_data, validation_data=None):
        """Train the estimator on the given data.

        Args:
            training_data (gluonts.dataset.common.Dataset): Dataset to train the model on.
            validation_data (gluonts.dataset.common.Dataset, optional): Dataset to validate the model on during training. Defaults to None.

        Returns:
            Predictor containing the trained model.
        """
        trained_models = []
        logger.info("Training one AutoARIMA model per time series ...")
        for item in tqdm(training_data):
            if self.season_length > 1:
                self._check_season_length(self.season_length, item[TIMESERIES_KEYS.TARGET], self.kwargs)
            external_features = self._set_external_features(self.kwargs, item)

            with threadpool_limits(limits=self.thread_limit, user_api="blas"):
                # calls to blas implementation will be limited to use only one thread
                model = pm.auto_arima(item[TIMESERIES_KEYS.TARGET], X=external_features, m=self.season_length, **self.kwargs)

            trained_models += [model]

        return AutoARIMAPredictor(prediction_length=self.prediction_length, freq=self.freq, trained_models=trained_models)

    def _check_season_length(self, season_length, target, kwargs):
        """Check if season_length is a working value for seasonality by performing the same test of seasonality pm.auto_arima does.
        The goal is for pm.auto_arima not to fail the training later.

        Args:
            season_length (int): Season length, always > 1.
            target (numpy.array): Target to train on.
            kwargs (dict): Kwargs dictionary of pm.auto_arima

        """
        logger.info(f"Check if seasonality 'm' can be set to {season_length}")
        try:
            nsdiffs(
                x=target.copy(),
                m=season_length,
                test=kwargs.get("seasonal_test", "ocsb"),
                max_D=kwargs.get("max_D", 1),
                **kwargs.get("seasonal_test_args", dict()),
            )
        except Exception as e:
            raise ValueError(f"Seasonality of AutoARIMA can't be set to {season_length}. Error when testing seasonality with nsdiffs: {e}")

    def _set_external_features(self, kwargs, item):
        external_features = None
        if self.use_feat_dynamic_real:
            external_features = item[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL].T
            logger.info("Using external features")
        return external_features
