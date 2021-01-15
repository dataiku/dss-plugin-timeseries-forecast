from gluonts_forecasts.timeseries_preparation import TimeseriesPreparator
import pandas as pd
import pytest


def test_duplicate_dates():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:12:00",
                "2021-01-01 17:35:00",
                "2021-01-02 14:55:00",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "D"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
    )
    with pytest.raises(ValueError):
        dataframe_prepared = preparator._truncate_dates(df)


def test_minutes_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:17:42",
                "2021-01-01 12:30:00",
                "2021-01-01 12:46:00",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "15min"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-01  12:15:00")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-01-01 12:45:00")


def test_hour_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-07 12:12:00",
                "2020-01-07 17:35:00",
                "2020-01-07 14:55:00",
                "2020-01-07 18:06:00",
                "2020-01-08 04:40:00",
                "2020-01-08 06:13:00",
                "2020-01-08 03:23:00",
            ],
            "id": [1, 1, 1, 1, 2, 2, 2],
        }
    )
    frequency = "2H"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
        max_timeseries_length=2,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)
    dataframe_prepared = preparator._keep_last_dates(dataframe_prepared)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-01-07 16:00:00")
    assert dataframe_prepared[time_column_name][3] == pd.Timestamp("2020-01-08 06:00:00")


def test_day_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:17:42",
                "2021-01-02 00:00:00",
                "2021-01-03 12:46:00",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "D"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-01")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-01-03")


def test_business_day_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-04 12:17:42",
                "2021-01-07 00:00:00",
                "2021-01-12 12:46:00",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "3B"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-04")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-01-07")


def test_week_sunday_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-03 12:12:00",
                "2021-01-05 17:35:00",
                "2021-01-15 14:55:00",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "W-SUN"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
        max_timeseries_length=2,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    dataframe_prepared = preparator._keep_last_dates(dataframe_prepared)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-10")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-01-17")


def test_quarter_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-15",
                "2021-03-28",
                "2021-06-11",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "3M"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-06-30")


def test_semester_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-15",
                "2021-06-28",
                "2021-12-01",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "6M"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-06-30")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-12-31")


def test_year_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-31",
                "2021-12-15",
                "2022-12-01",
            ],
            "id": [1, 1, 1],
        }
    )
    frequency = "12M"
    time_column_name = "date"
    timeseries_identifiers_names = ["id"]
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(
        time_column_name=time_column_name,
        frequency=frequency,
        timeseries_identifiers_names=timeseries_identifiers_names,
    )
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-12-31")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2022-12-31")