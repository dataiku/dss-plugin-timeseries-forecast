from gluonts_forecasts.training_session import TrainingSession
from dku_constants import METRICS_DATASET
from datetime import datetime
import pandas as pd


class TestCrossValidation:
    def setup_class(self):
        self.df = pd.DataFrame(
            {
                "date": [
                    "2020-01-12 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-12 12:00:00",
                    "2020-01-12 18:00:00",
                    "2020-01-13 00:00:00",
                    "2020-01-13 06:00:00",
                    "2020-01-13 12:00:00",
                    "2020-01-13 18:00:00",
                    "2020-01-12 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-12 12:00:00",
                    "2020-01-12 18:00:00",
                    "2020-01-13 00:00:00",
                    "2020-01-13 06:00:00",
                    "2020-01-13 12:00:00",
                    "2020-01-13 18:00:00",
                ],
                "target": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
                "item": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            }
        )
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.tz_localize(tz=None)
        self.models_parameters = {
            "trivial_identity": {"activated": True, "method": "trivial_identity", "kwargs": {"num_samples": 100}},
        }
        self.session_name = datetime.utcnow().isoformat() + "Z"
        self.target_length = 8

    def setup_method(self):
        self.training_session = TrainingSession(
            target_columns_names=["target"],
            time_column_name="date",
            frequency="6H",
            epoch=2,
            models_parameters=self.models_parameters,
            prediction_length=2,
            training_df=self.df,
            make_forecasts=True,
            external_features_columns_names=None,
            timeseries_identifiers_names=["item"],
            batch_size=32,
            user_num_batches_per_epoch=-1,
            timeseries_cross_validation=True,
            rolling_windows_number=3,
            cutoff_period=1,
        )
        self.training_session.init(self.session_name)
        self.training_session.create_gluon_list_datasets()
        self.training_session.instantiate_models()
        self.training_session.train_evaluate()

    def test_cut_lengths_train_test_pairs(self):
        expected_cut_lengths_train_test_pairs = [(4, 2), (3, 1), (2, 0)]
        assert (
            self.training_session.rolling_windows_cut_lengths_train_test_pairs == expected_cut_lengths_train_test_pairs
        )

    def test_gluon_list_datasets_by_cut_length(self):
        for cut_length, gluon_list_dataset in self.training_session.gluon_list_datasets_by_cut_length.items():
            assert len(gluon_list_dataset.list_data[0].get("target")) == self.target_length - cut_length

    def test_metrics_df(self):
        metrics_df = self.training_session.get_evaluation_metrics_to_display()
        # metrics has 9 rows = 1 model * 1 overall aggregated rows + 1 model * 2 timeseries * 1 rolling windows aggregated row
        # + 1 model * 2 timeseries * 3 rolling windows
        assert metrics_df.shape == (9, 15)
        assert len(metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW].index) == 1

        rolling_windows_metrics = metrics_df[
            metrics_df[METRICS_DATASET.ROLLING_WINDOWS] != METRICS_DATASET.AGGREGATED_ROW
        ]
        assert set(rolling_windows_metrics[METRICS_DATASET.ROLLING_WINDOWS].unique()) == set((0, 1, 2))

        for identifier in [1, 2]:
            identifier_df = metrics_df[metrics_df["item"] == identifier]
            aggregation = identifier_df[
                identifier_df[METRICS_DATASET.ROLLING_WINDOWS] == METRICS_DATASET.AGGREGATED_ROW
            ]["mape"].iloc[0]
            average = identifier_df[identifier_df[METRICS_DATASET.ROLLING_WINDOWS] != METRICS_DATASET.AGGREGATED_ROW][
                "mape"
            ].mean()
            assert round(aggregation, 4) == round(average, 4)
