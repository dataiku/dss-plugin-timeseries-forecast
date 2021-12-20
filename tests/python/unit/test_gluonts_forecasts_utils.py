from gluonts.dataset.common import ListDataset
from timeseries_preparation.preparation import assert_time_column_valid
from gluonts_forecasts.utils import add_future_external_features
from dku_constants import TIMESERIES_KEYS
import pandas as pd
import numpy as np
import pytest


def test_valid_minute_frequency():
    time_column_name = "date"
    frequency = "10min"
    df = pd.DataFrame(
        {
            time_column_name: [
                "2020-01-31 06:30:30",
                "2020-01-31 06:40:30",
                "2020-01-31 06:50:30",
                "2020-01-31 07:00:30",
            ]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    try:
        assert_time_column_valid(df, time_column_name, frequency)
    except ValueError as e:
        pytest.fail("Unexpected ValueError: {}".format(e))


def test_invalid_minute_frequency():
    time_column_name = "date"
    frequency = "10min"
    df = pd.DataFrame(
        {
            time_column_name: [
                "2020-01-31 06:30:30",
                "2020-01-31 06:40:31",
                "2020-01-31 06:50:30",
                "2020-01-31 07:00:30",
            ]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    with pytest.raises(ValueError):
        assert_time_column_valid(df, time_column_name, frequency)


def test_valid_business_day_frequency():
    time_column_name = "date"
    frequency = "2B"
    df = pd.DataFrame(
        {
            time_column_name: [
                "2020-01-01 06:30:00",
                "2020-01-03 06:30:00",
                "2020-01-07 06:30:00",
                "2020-01-09 06:30:00",
            ]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    try:
        assert_time_column_valid(df, time_column_name, frequency)
    except ValueError as e:
        pytest.fail("Unexpected ValueError: {}".format(e))


def test_valid_month_frequency():
    time_column_name = "date"
    frequency = "2M"
    df = pd.DataFrame(
        {
            time_column_name: [
                "2020-02-29 00:00:00",
                "2020-04-30 00:00:00",
                "2020-06-30 00:00:00",
                "2020-08-31 00:00:00",
            ]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    try:
        assert_time_column_valid(df, time_column_name, frequency)
    except ValueError as e:
        pytest.fail("Unexpected ValueError: {}".format(e))


def test_invalid_month_frequency():
    time_column_name = "date"
    frequency = "6M"
    df = pd.DataFrame(
        {
            time_column_name: [
                "2018-01-31 00:00:00",
                "2018-07-31 12:30:00",
                "2019-01-30 00:00:00",
                "2019-07-31 00:00:00",
            ]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    with pytest.raises(ValueError):
        assert_time_column_valid(df, time_column_name, frequency)


def test_valid_year_frequency():
    time_column_name = "date"
    frequency = "1Y"
    df = pd.DataFrame({time_column_name: ["2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"]})
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    try:
        assert_time_column_valid(df, time_column_name, frequency)
    except ValueError as e:
        pytest.fail("Unexpected ValueError: {}".format(e))


def test_invalid_year_frequency():
    time_column_name = "date"
    frequency = "2Y"
    df = pd.DataFrame({time_column_name: ["2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01"]})
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    with pytest.raises(ValueError):
        assert_time_column_valid(df, time_column_name, frequency)


def test_add_future_external_features_with_identifiers():
    frequency = "D"
    timeseries_0 = {
        TIMESERIES_KEYS.START: "2018-01-01",
        TIMESERIES_KEYS.TARGET: np.array([12, 13, 14, 15, 16]),
        TIMESERIES_KEYS.TARGET_NAME: "sales_0",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 1]]),
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
        TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 1},
    }
    timeseries_1 = {
        TIMESERIES_KEYS.START: "2018-01-01",
        TIMESERIES_KEYS.TARGET: np.array([2, 3, 4, 5, 6]),
        TIMESERIES_KEYS.TARGET_NAME: "sales_1",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]),
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
        TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 2},
    }
    gluon_train_dataset = ListDataset([timeseries_0, timeseries_1], freq=frequency)

    external_features_future_df = pd.DataFrame(
        {
            "date": [
                "2018-01-06",
                "2018-01-07",
                "2018-01-08",
                "2018-01-06",
                "2018-01-07",
                "2018-01-08",
            ],
            "store": [1, 1, 1, 1, 1, 1],
            "item": [1, 1, 1, 2, 2, 2],
            "is_holiday": [0, 0, 0, 0, 1, 0],
            "is_weekend": [1, 0, 0, 1, 0, 0],
        }
    )
    external_features_future_df["date"] = pd.to_datetime(external_features_future_df["date"]).dt.tz_localize(tz=None)

    prediction_length = 3
    gluon_dataset = add_future_external_features(
        gluon_train_dataset, external_features_future_df, prediction_length, frequency
    )

    assert (
        gluon_dataset.list_data[0][TIMESERIES_KEYS.FEAT_DYNAMIC_REAL]
        == np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])
    ).all()


def test_invalid_add_future_external_features_with_identifiers():
    frequency = "D"
    timeseries_0 = {
        TIMESERIES_KEYS.START: "2018-01-01",
        TIMESERIES_KEYS.TARGET: np.array([12, 13, 14, 15, 16]),
        TIMESERIES_KEYS.TARGET_NAME: "sales_0",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 1]]),
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
        TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 1},
    }
    timeseries_1 = {
        TIMESERIES_KEYS.START: "2018-01-01",
        TIMESERIES_KEYS.TARGET: np.array([2, 3, 4, 5, 6]),
        TIMESERIES_KEYS.TARGET_NAME: "sales_1",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL: np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]),
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES: ["is_holiday", "is_weekend"],
        TIMESERIES_KEYS.IDENTIFIERS: {"store": 1, "item": 2},
    }
    gluon_train_dataset = ListDataset([timeseries_0, timeseries_1], freq=frequency)

    external_features_future_df = pd.DataFrame(
        {
            "date": [
                "2018-01-06",
                "2018-01-07",
                "2018-01-08",
            ],
            "store": [1, 1, 1],
            "item": [1, 1, 1],
            "is_holiday": [0, 0, 0],
            "is_weekend": [1, 0, 0],
        }
    )
    external_features_future_df["date"] = pd.to_datetime(external_features_future_df["date"]).dt.tz_localize(tz=None)

    prediction_length = 3

    with pytest.raises(ValueError):
        _ = add_future_external_features(gluon_train_dataset, external_features_future_df, prediction_length, frequency)
