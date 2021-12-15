from gluonts.dataset.common import ListDataset
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts_forecasts.trained_model import TrainedModel
from dku_constants import TIMESERIES_KEYS, METRICS_DATASET, ROW_ORIGIN
from datetime import datetime
import pandas as pd
import numpy as np


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

        train_timeseries_0 = timeseries_0.copy()
        train_timeseries_0[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])

        train_timeseries_1 = timeseries_1.copy()
        train_timeseries_1[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])

        self.train_gluon_dataset = ListDataset([train_timeseries_0, train_timeseries_1], freq=self.frequency)

        deepar_estimator = DeepAREstimator(
            freq=self.frequency,
            prediction_length=self.prediction_length,
            trainer=Trainer(epochs=1),
            use_feat_dynamic_real=True,
        )

        deepar_predictor = deepar_estimator.train(self.train_gluon_dataset)

        self.predictors = {
            "TrivialIdentity": IdentityPredictor(
                prediction_length=self.prediction_length, freq=self.frequency, num_samples=100
            ),
            "SeasonalNaive": SeasonalNaivePredictor(
                freq=self.frequency, prediction_length=self.prediction_length, season_length=2
            ),
            "DeepAR": deepar_predictor,
        }

    def setup_method(self):
        self.session_name = datetime.utcnow().isoformat() + "Z"
        self.trained_model = TrainedModel(
            gluon_dataset=self.gluon_dataset,
            prediction_length=self.prediction_length,
            frequency=self.frequency,
            quantiles=[0.1, 0.5, 0.9],
            include_history=True,
        )

    def test_predict(self):
        model_label = "TrivialIdentity"
        forecasts_df = self.trained_model.predict(model_label, self.predictors[model_label])
        forecasts_df = self.trained_model.get_forecasts_df_for_display(forecasts_df, session=self.session_name)
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
            ROW_ORIGIN.COLUMN_NAME,
        ]
        assert set(expected_forecasts_columns) == set(forecasts_df.columns)

        future_df = forecasts_df[forecasts_df["item"] == 2].iloc[0:2]
        history_df = forecasts_df[forecasts_df["item"] == 2].iloc[2:]

        assert future_df["sales"].count() == 0 and history_df["sales"].count() == 4
        assert future_df["forecast_sales"].count() == 2 and history_df["forecast_sales"].count() == 0

    def test_forecast_all_models(self):
        forecasts_df = pd.DataFrame()
        for model_label, predictor in self.predictors.items():
            single_forecasts_df = self.trained_model.predict(model_label, predictor)
            forecasts_df = forecasts_df.append(single_forecasts_df)
        forecasts_df = self.trained_model.get_forecasts_df_for_display(forecasts_df, self.session_name)

        assert len(forecasts_df.index) == 20
        assert len(forecasts_df[forecasts_df[ROW_ORIGIN.COLUMN_NAME] == ROW_ORIGIN.FORECAST].index) == 12
        assert len(forecasts_df[forecasts_df[ROW_ORIGIN.COLUMN_NAME] == ROW_ORIGIN.HISTORY].index) == 8
        assert set(forecasts_df[METRICS_DATASET.MODEL_COLUMN].unique()) == set(
            {"DeepAR", "SeasonalNaive", "TrivialIdentity", np.nan}
        )
