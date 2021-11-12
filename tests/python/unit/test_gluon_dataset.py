from gluonts_forecasts.gluon_dataset import GluonDataset
from dku_constants import TIMESERIES_KEYS
import pandas as pd
import numpy as np


class TestGluonDataset:
    def setup_class(self):
        self.df = pd.DataFrame(
            {
                "date": ["2018-01-06", "2018-01-07", "2018-01-08", "2018-01-06", "2018-01-07", "2018-01-08"],
                "volume": [2, 4, 2, 5, 2, 5],
                "revenue": [12, 13, 14, 15, 11, 10],
                "store": [1, 1, 1, 1, 1, 1],
                "item": [1, 1, 1, 2, 2, 2],
                "is_holiday": [0, 0, 0, 0, 1, 0],
                "is_weekend": [1, 0, 0, 1, 0, 0],
            }
        )
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.tz_localize(tz=None)

    def setup_method(self):
        self.gluon_dataset = GluonDataset(
            time_column_name="date",
            frequency="D",
            target_columns_names=["volume", "revenue"],
            timeseries_identifiers_names=["store", "item"],
            external_features_columns_names=["is_holiday", "is_weekend"],
            min_length=2,
        )
        self.gluon_list_dataset = self.gluon_dataset.create_list_datasets(self.df)[0]

    def test_start_date(self):
        assert self.gluon_list_dataset.list_data[1][TIMESERIES_KEYS.START] == pd.Timestamp("2018-01-06")

    def test_target(self):
        assert (self.gluon_list_dataset.list_data[1][TIMESERIES_KEYS.TARGET] == np.array([12, 13, 14])).all()

    def test_external_features(self):
        assert (
            self.gluon_list_dataset.list_data[1][TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] == np.array([[0, 0, 0], [1, 0, 0]])
        ).all()

    def test_timeseries_identifiers(self):
        assert self.gluon_list_dataset.list_data[2][TIMESERIES_KEYS.IDENTIFIERS] == {"store": 1, "item": 2}
