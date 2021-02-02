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


logger = SafeLogger("Forecast plugin - STL")


class STLPredictor(RepresentablePredictor):
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
    def __init__(self, prediction_length, freq, models, lead_time=0):
        super().__init__(freq=freq, lead_time=lead_time, prediction_length=prediction_length)
        self.models = models

    def predict(self, dataset, **kwargs):
        """

        Args:
            dataset (gluonts.dataset.common.Dataset): Dataset after wich to predict forecasts.

        Yields:
            SampleForecast of predictions.
        """
        logger.info("Training models and predicting time series ...")
        for i, item in tqdm(enumerate(dataset)):
            yield self.predict_item(item, self.models[i])

    def predict_item(self, item, model):
        """Compute quantiles using the confidence intervals of autoarima.

        Args:
            item (DataEntry): One timeseries.
            model (list): List of STL instanciated models.

        Returns:
            SampleForecast of quantiles.
        """
        target_length = len(item[TIMESERIES_KEYS.TARGET])
        start_date = frequency_add(item[TIMESERIES_KEYS.START], target_length)

        # need to fit the model in the Predictor class because the results of model.fit() is some un-pickable cython
        trained_model = model.fit()

        samples = []
        for alpha in np.arange(0.02, 1.01, 0.02):
            predictions = trained_model.get_prediction(start=target_length, end=target_length + self.prediction_length - 1)
            confidence_intervals = predictions.conf_int(alpha=alpha)
            samples += [confidence_intervals["lower"].values, confidence_intervals["upper"].values]

        return SampleForecast(samples=np.stack(samples), start_date=start_date, freq=self.freq)


class STLEstimator(Estimator):
    @validated()
    def __init__(self, prediction_length, freq, season_length=None, **kwargs):
        super().__init__()
        self.prediction_length = prediction_length
        self.freq = freq
        self.kwargs = cast_kwargs(kwargs)
        if "period" in kwargs:
            raise ValueError("Keyword argument 'period' is not writable for STL, please use the Seasonality parameter")
        self.season_length = season_length if season_length is not None else 1

    def train(self, training_data):
        """Train the estimator on the given data.

        Args:
            training_data (gluonts.dataset.common.Dataset): Dataset to train the model on.

        Returns:
            Predictor containing the instanciated STL models.
        """
        models = []
        logger.info("Creating one STL model per time series ...")
        for item in tqdm(training_data):
            model = STLForecast(endog=pd.Series(item[TIMESERIES_KEYS.TARGET]), model=ETSModel, period=self.season_length, **self.kwargs)
            models += [model]

        return STLPredictor(prediction_length=self.prediction_length, freq=self.freq, models=models)
