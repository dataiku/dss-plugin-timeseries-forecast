from gluonts_forecasts.timeseries_preparation import TimeseriesPreparator
import pandas as pd


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
    preparator = TimeseriesPreparator(time_column_name, frequency, timeseries_identifiers_names)
    dataframe_prepared = preparator.truncate_timestamps(df)
    dataframe_prepared = preparator.sort(dataframe_prepared)
    preparator.check_regular_frequency(dataframe_prepared)
    dataframe_prepared = preparator.keep_last_timestamps(dataframe_prepared, 2)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-01-07 16:00:00")
    assert dataframe_prepared[time_column_name][3] == pd.Timestamp("2020-01-08 06:00:00")


def test_week_sunday_truncation():
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:12:00",
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
    preparator = TimeseriesPreparator(time_column_name, frequency, timeseries_identifiers_names)
    dataframe_prepared = preparator.truncate_timestamps(df)
    dataframe_prepared = preparator.sort(dataframe_prepared)
    preparator.check_regular_frequency(dataframe_prepared)

    dataframe_prepared = preparator.keep_last_timestamps(dataframe_prepared, 2)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-10")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-01-17")