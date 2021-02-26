from gluonts.dataset.common import ListDataset
from gluonts_forecasts.model import Model
from timeseries_preparation.preparation import TimeseriesPreparator
from gluonts_forecasts.gluon_dataset import GluonDataset
from gluonts_forecasts.model_handler import MODEL_DESCRIPTORS, LABEL, list_available_models
from dku_constants import METRICS_DATASET, EVALUATION_METRICS_DESCRIPTIONS, TIMESERIES_KEYS
from datetime import datetime
from pandas.api.types import is_datetime64_ns_dtype
import pandas as pd
import numpy as np
import pytest


def test_missing_values_target():
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            {
                "date": ["2018-01-06", "2018-01-07", "2018-01-08", "2018-01-06", "2018-01-07", "2018-01-08"],
                "volume": [2, 4, np.NaN, 5, 2, 5],
                "item": [1, 1, 1, 2, 2, 2],
            }
        )

        timeseries_preparator = TimeseriesPreparator(
            time_column_name="date",
            frequency="D",
            target_columns_names=["volume"],
            timeseries_identifiers_names=["item"],
        )

        training_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)


def test_missing_values_identifiers():
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            {
                "date": ["2018-01-06", "2018-01-07", "2018-01-08", "2018-01-06", "2018-01-07", "2018-01-08"],
                "volume": [2, 4, 2, 5, 2, 5],
                "item": [1, 1, np.NaN, 2, 2, 2],
            }
        )

        timeseries_preparator = TimeseriesPreparator(
            time_column_name="date",
            frequency="D",
            target_columns_names=["volume"],
            timeseries_identifiers_names=["item"],
        )

        training_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)


def test_missing_values_external_features():
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            {
                "date": ["2018-01-06", "2018-01-07", "2018-01-08", "2018-01-06", "2018-01-07", "2018-01-08"],
                "volume": [2, 4, 2, 5, 2, 5],
                "item": [1, 1, 1, 2, 2, 2],
                "ext_feat": [1, 0, np.NaN, 0, 0, 1],
            }
        )

        timeseries_preparator = TimeseriesPreparator(
            time_column_name="date",
            frequency="D",
            target_columns_names=["volume"],
            timeseries_identifiers_names=["item"],
            external_features_columns_names=["ext_feat"],
        )

        training_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)
