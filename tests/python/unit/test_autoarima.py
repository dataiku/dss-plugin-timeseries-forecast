from gluonts_forecasts.custom_models.autoarima import AutoARIMAEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from dku_constants import TIMESERIES_KEYS
from gluonts.dataset.common import ListDataset
import numpy as np
import pytest


class TestAutoARIMATrain:
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

    def test_training_hour_frequency(self):
        prediction_length = 2
        frequency = "12H"
        gluon_dataset = ListDataset(self.timeseries, freq=frequency)
        estimator = AutoARIMAEstimator(prediction_length=prediction_length, freq=frequency, season_length=2)
        predictor = estimator.train(gluon_dataset)

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[1].samples.shape == (100, 2)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MSE"] is not None

    def test_training_month_frequency(self):
        prediction_length = 1
        frequency = "3M"
        gluon_dataset = ListDataset([self.timeseries[0]], freq=frequency)
        estimator = AutoARIMAEstimator(prediction_length=prediction_length, freq=frequency, max_D=3, season_length=4)
        predictor = estimator.train(gluon_dataset)

        assert predictor.trained_models[0].seasonal_order[3] == 4  # seasonality is 4

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[0].samples.shape == (100, 1)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MSE"] is not None

    def test_training_bad_seasonality_user_input(self):
        prediction_length = 1
        frequency = "3M"
        gluon_dataset = ListDataset(self.timeseries, freq=frequency)
        with pytest.raises(ValueError):
            estimator = AutoARIMAEstimator(prediction_length=prediction_length, freq=frequency, m=12)

    def test_training_external_features(self):
        prediction_length = 2
        frequency = "3M"
        gluon_dataset = ListDataset(self.timeseries, freq=frequency)
        estimator = AutoARIMAEstimator(
            prediction_length=prediction_length, freq=frequency, season_length=4, use_feat_dynamic_real=True
        )
        predictor = estimator.train(gluon_dataset)

        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        assert forecasts[1].samples.shape == (100, 2)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(gluon_dataset))
        assert agg_metrics["MAPE"] is not None
