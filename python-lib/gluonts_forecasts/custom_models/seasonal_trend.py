from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import frequency_add
from gluonts.core.component import validated
from gluonts_forecasts.custom_models.utils import cast_kwargs
from constants import TIMESERIES_KEYS
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
import pandas as pd
from safe_logger import SafeLogger
from tqdm import tqdm


logger = SafeLogger("Forecast plugin - SeasonalTrend")


class SeasonalTrendPredictor(RepresentablePredictor):
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
        logger.info("Training models and predicting time series ...")
        for i, item in tqdm(enumerate(dataset)):
            yield self.predict_item(item, self.trained_models[i])

    def predict_item(self, item, trained_model):
        """Compute quantiles using the confidence intervals of autoarima.

        Args:
            item (DataEntry): One timeseries.
            trained_model (STLForecastResults): Trained STL model.

        Returns:
            SampleForecast of quantiles.
        """
        target_length = len(item[TIMESERIES_KEYS.TARGET])
        start_date = frequency_add(item[TIMESERIES_KEYS.START], target_length)

        samples = []
        for alpha in np.arange(0.02, 1.01, 0.02):
            predictions = trained_model.get_prediction(start=target_length, end=target_length + self.prediction_length - 1)
            confidence_intervals = predictions.conf_int(alpha=alpha)
            samples += [confidence_intervals["lower"].values, confidence_intervals["upper"].values]

        return SampleForecast(samples=np.stack(samples), start_date=start_date, freq=self.freq)


class SeasonalTrendEstimator(Estimator):
    @validated()
    def __init__(self, prediction_length, freq, season_length=None, **kwargs):
        super().__init__()
        self.prediction_length = prediction_length
        self.freq = freq
        self.kwargs = cast_kwargs(kwargs)
        if "period" in self.kwargs:
            raise ValueError("Keyword argument 'period' is not writable for STL, please use the Seasonality parameter")
        if "model" not in self.kwargs:
            self.kwargs["model"] = ETSModel
        self.season_length = season_length if season_length is not None else 1

    def train(self, training_data):
        """Train the estimator on the given data.

        Args:
            training_data (gluonts.dataset.common.Dataset): Dataset to train the model on.

        Returns:
            Predictor containing the trained STL models.
        """
        trained_models = []
        logger.info("Creating one SeasonalTrend model per time series ...")
        for item in tqdm(training_data):
            model = STLForecast(endog=pd.Series(item[TIMESERIES_KEYS.TARGET]), period=self.season_length, **self.kwargs)
            trained_model = model.fit()
            trained_models += [trained_model]

        return SeasonalTrendPredictor(prediction_length=self.prediction_length, freq=self.freq, trained_models=trained_models)
