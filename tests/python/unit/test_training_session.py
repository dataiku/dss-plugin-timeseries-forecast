from gluonts_forecasts.training_session import TrainingSession
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from dku_constants import TIMESERIES_KEYS, METRICS_DATASET, EVALUATION_METRICS_DESCRIPTIONS, ROW_ORIGIN
from datetime import datetime
import pandas as pd


class TestTrainingSession:
    def setup_class(self):
        self.df = pd.DataFrame(
            {
                "date": [
                    "2020-01-12 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-12 12:00:00",
                    "2020-01-12 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-12 12:00:00",
                ],
                "volume": [2, 4, 2, 5, 2, 5],
                "revenue": [12, 13, 14, 15, 11, 10],
                "store": [1, 1, 1, 1, 1, 1],
                "item": [1, 1, 1, 2, 2, 2],
                "is_holiday": [0, 0, 0, 0, 1, 0],
                "is_weekend": [1, 0, 0, 1, 0, 0],
            }
        )
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.tz_localize(tz=None)
        self.models_parameters = {
            "deepar": {"activated": True, "kwargs": {"dropout_rate": "0.3", "cell_type": "gru"}},
            "mqcnn": {"activated": True, "kwargs": {}},
            "trivial_identity": {"activated": True, "method": "trivial_identity", "kwargs": {"num_samples": 100}},
        }
        self.session_name = datetime.utcnow().isoformat() + "Z"

    def setup_method(self):
        self.training_session = TrainingSession(
            target_columns_names=["volume", "revenue"],
            time_column_name="date",
            frequency="6H",
            epoch=1,
            models_parameters=self.models_parameters,
            prediction_length=1,
            training_df=self.df,
            make_forecasts=True,
            external_features_columns_names=["is_holiday", "is_weekend"],
            timeseries_identifiers_names=["store", "item"],
            batch_size=32,
            user_num_batches_per_epoch=-1,
        )
        self.training_session.init(self.session_name)
        self.training_session.create_gluon_list_datasets()
        self.training_session.instantiate_models()

    def test_gluon_list_datasets(self):
        test_timeseries_length = len(self.training_session.full_list_dataset.list_data[0][TIMESERIES_KEYS.TARGET])
        train_timeseries_length = len(
            self.training_session.evaluation_train_list_dataset.list_data[0][TIMESERIES_KEYS.TARGET]
        )
        assert test_timeseries_length == train_timeseries_length + self.training_session.prediction_length
        assert self.training_session.num_batches_per_epoch == 50

    def test_evaluation_metrics(self):
        self.training_session.train_evaluate()
        expected_metrics_columns = ["store", "item"]
        expected_metrics_columns += [
            METRICS_DATASET.TARGET_COLUMN,
            METRICS_DATASET.MODEL_COLUMN,
            METRICS_DATASET.MODEL_PARAMETERS,
            METRICS_DATASET.TRAINING_TIME,
            METRICS_DATASET.SESSION,
        ]
        expected_metrics_columns += list(EVALUATION_METRICS_DESCRIPTIONS.keys())
        metrics_models = self.training_session.metrics_df[METRICS_DATASET.MODEL_COLUMN].unique()
        model_config_registry = ModelConfigRegistry()
        expected_metrics_models = [
            model_config_registry.get_model("deepar").get_label(),
            model_config_registry.get_model("mqcnn").get_label(),
            model_config_registry.get_model("trivial_identity").get_label(),
        ]
        assert len(self.training_session.metrics_df.index) == 15
        assert set(self.training_session.metrics_df.columns) == set(expected_metrics_columns)
        assert set(metrics_models) == set(expected_metrics_models)

    def test_evaluation_forecasts(self):
        self.training_session.train_evaluate(retrain=True)
        expected_evaluation_forecasts_columns = [
            "date",
            "volume",
            "revenue",
            "store",
            "item",
            "is_holiday",
            "is_weekend",
            "deepar_volume",
            "deepar_revenue",
            "mqcnn_volume",
            "mqcnn_revenue",
            "trivial_identity_volume",
            "trivial_identity_revenue",
            METRICS_DATASET.SESSION,
            ROW_ORIGIN.COLUMN_NAME,
        ]
        not_nan_count = self.training_session.evaluation_forecasts_df.count()
        assert len(self.training_session.evaluation_forecasts_df.index) == 6
        assert set(self.training_session.evaluation_forecasts_df.columns) == set(expected_evaluation_forecasts_columns)
        assert not_nan_count["volume"] == 6 and not_nan_count["deepar_volume"] == 2

    def test_retrain(self):
        self.training_session.train_evaluate(retrain=True)
