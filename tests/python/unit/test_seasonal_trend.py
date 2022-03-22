from gluonts_forecasts.custom_models.seasonal_trend import SeasonalTrendEstimator
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from dku_constants import TIMESERIES_KEYS
from gluonts.dataset.common import ListDataset
import numpy as np


class TestSeasonalTrendTrain:
    def setup_class(self):
        self.timeseries = [
            {
                TIMESERIES_KEYS.START: "2021-01-15 00:00:00",
                TIMESERIES_KEYS.TARGET: np.array(
                    [1, 1, 2, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 1, 3, 4, 1, 2, 3, 4, 1, 3, 4, 2, 3, 3, 2]
                ),
                TIMESERIES_KEYS.TARGET_NAME: "target_1",
                TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
                TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array(
                    [[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 4, 1, 0, 0, 4, 1, 0, 4, 0, 0, 0, 0]]
                ),
                TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: "ext_feat",
            },
            {
                TIMESERIES_KEYS.START: "2021-01-18 00:00:00",
                TIMESERIES_KEYS.TARGET: np.array(
                    [1, 3, 2, 3, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 0, 0, 0]
                ),
                TIMESERIES_KEYS.TARGET_NAME: "target_2",
                TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
                TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array(
                    [[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]]
                ),
                TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: "ext_feat",
            },
        ]

    def test_training_hour_frequency_ets(self):
        prediction_length = 2
        frequency = "12H"
        gluon_dataset = ListDataset(self.timeseries, freq=frequency)
        kwargs = {"model": ETSModel, "model_kwargs": {"seasonal_periods": 13}}
        estimator = SeasonalTrendEstimator(
            prediction_length=prediction_length, freq=frequency, season_length=2, **kwargs
        )
        predictor = estimator.train(gluon_dataset)

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[1].samples.shape == (100, 2)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MSE"] is not None

    def test_training_month_frequency_arima(self):
        prediction_length = 1
        frequency = "3M"
        gluon_dataset = ListDataset([self.timeseries[0]], freq=frequency)
        kwargs = {"model": ARIMA, "model_kwargs": {"order": (2, 1, 1)}}
        estimator = SeasonalTrendEstimator(
            prediction_length=prediction_length, freq=frequency, season_length=4, **kwargs
        )
        predictor = estimator.train(gluon_dataset)

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[0].samples.shape == (100, 1)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MSE"] is not None

    def test_training_long_format(self):
        prediction_length = 2
        frequency = "3M"
        self.timeseries[0][TIMESERIES_KEYS.IDENTIFIERS] = {"country": "uk"}
        self.timeseries[1][TIMESERIES_KEYS.IDENTIFIERS] = {"country": "usa"}

        gluon_dataset = ListDataset(self.timeseries, freq=frequency)
        kwargs = {"model": ARIMA, "model_kwargs": {"order": (2, 1, 1)}}
        estimator = SeasonalTrendEstimator(
            prediction_length=prediction_length, freq=frequency, season_length=4, **kwargs
        )
        predictor = estimator.train(gluon_dataset)

        assert frozenset({"country": "uk", TIMESERIES_KEYS.TARGET_NAME: "target_1"}.items()) in predictor.trained_models
        assert frozenset({"country": "usa", TIMESERIES_KEYS.TARGET_NAME: "target_2"}.items()) in predictor.trained_models

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[0].samples.shape == (100, 2)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MSE"] is not None
