from gluonts.dataset.common import ListDataset
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts_forecasts.trained_model import TrainedModel
from constants import TIMESERIES_KEYS, METRICS_DATASET
from datetime import datetime
import pandas as pd
import numpy as np
import pytest


class TestTrainedModel:
    def setup_class(self):
        self.frequency = "D"
        self.prediction_length = 2
        timeseries_0 = {
            TIMESERIES_KEYS.START: "2018-01-01",
            TIMESERIES_KEYS.TARGET: np.array([12, 13, 14, 15]),
            TIMESERIES_KEYS.TARGET_NAME: "sales",
            TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
            TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 0]]),
            TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
            TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 1},
        }
        timeseries_1 = {
            TIMESERIES_KEYS.START: "2018-01-01",
            TIMESERIES_KEYS.TARGET: np.array([2, 3, 4, 5]),
            TIMESERIES_KEYS.TARGET_NAME: "sales",
            TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
            TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 0]]),
            TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
            TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 2},
        }
        self.gluon_dataset = ListDataset([timeseries_0, timeseries_1], freq=self.frequency)
        self.predictor = IdentityPredictor(prediction_length=2, freq="D", num_samples=100)

    def setup_method(self):
        self.session_name = datetime.utcnow().isoformat() + "Z"
        self.model_label = "TrivialIdentity"
        self.trained_model = TrainedModel(
            predictor=self.predictor,
            gluon_dataset=self.gluon_dataset,
            prediction_length=self.prediction_length,
            quantiles=[0.1, 0.5, 0.9],
            include_history=True,
        )

    def test_predict(self):
        self.trained_model.predict()
        forecasts_df = self.trained_model.get_forecasts_df(session=self.session_name, model_label=self.model_label)
        expected_forecasts_columns = [
            "date",
            "store",
            "item",
            "is_holiday",
            "is_weekend",
            "sales",
            "forecast_lower_sales",
            "forecast_sales",
            "forecast_upper_sales",
            METRICS_DATASET.SESSION,
            METRICS_DATASET.MODEL_COLUMN,
        ]
        assert set(expected_forecasts_columns) == set(forecasts_df.columns)

        future_df = forecasts_df[forecasts_df["item"] == 2].iloc[0:2]
        history_df = forecasts_df[forecasts_df["item"] == 2].iloc[2:]

        assert future_df["sales"].count() == 0 and history_df["sales"].count() == 4
        assert future_df["forecast_sales"].count() == 2 and history_df["forecast_sales"].count() == 0
