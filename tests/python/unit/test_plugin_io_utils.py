from plugin_io_utils import check_continuous_time_column
from dku_timeseries.prediction import add_future_external_features 
from gluonts.dataset.common import ListDataset
import pandas as pd
import numpy as np

#dataset = {'time_column': ['', '', ''], 'data_column': [0, 1, 2]}
dataset_timeline_gap = {'time_column': ['2013-02-08T00:00:00.000Z', '2013-02-11T00:00:00.000Z', '2013-02-12T00:00:00.000Z'], 'data_column': [0, 1, 2]}
dataset_timeline = {'time_column': ['2013-02-10T00:00:00.000Z', '2013-02-11T00:00:00.000Z', '2013-02-12T00:00:00.000Z'], 'data_column': [0, 1, 2]}

dataframe_timeline = pd.DataFrame.from_dict(dataset_timeline)
dataframe_timeline['time_column'] = pd.to_datetime(dataframe_timeline['time_column'])
dataframe_timeline_gap = pd.DataFrame.from_dict(dataset_timeline_gap)
dataframe_timeline_gap['time_column'] = pd.to_datetime(dataframe_timeline_gap['time_column'])


def test_valid_minute_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'min'
    time_granularity_step = '10'
    df = pd.DataFrame(
        {time_column_name: [
            '2020-01-31 06:30:30',
            '2020-01-31 06:40:30',
            '2020-01-31 06:50:30',
            '2020-01-31 07:00:30']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is True


def test_invalid_minute_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'min'
    time_granularity_step = '10'
    df = pd.DataFrame(
        {time_column_name: [
            '2020-01-31 06:30:30',
            '2020-01-31 06:40:31',
            '2020-01-31 06:50:30',
            '2020-01-31 07:00:30']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is False


def test_valid_business_day_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'B'
    time_granularity_step = '2'
    df = pd.DataFrame(
        {time_column_name: [
            '2020-01-01 06:30:00',
            '2020-01-03 06:30:00',
            '2020-01-07 06:30:00',
            '2020-01-09 06:30:00']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is True


def test_valid_month_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'M'
    time_granularity_step = '2'
    df = pd.DataFrame(
        {time_column_name: [
            '2020-02-29 00:00:00',
            '2020-04-30 00:00:00',
            '2020-06-30 00:00:00',
            '2020-08-31 00:00:00']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is True


def test_invalid_month_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'M'
    time_granularity_step = '6'
    df = pd.DataFrame(
        {time_column_name: [
            '2018-01-31 00:00:00',
            '2018-07-31 12:30:00',
            '2019-01-30 00:00:00',
            '2019-07-31 00:00:00']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is False


def test_valid_year_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'Y'
    time_granularity_step = '1'
    df = pd.DataFrame(
        {time_column_name: [
            '2017-12-31',
            '2018-12-31',
            '2019-12-31',
            '2020-12-31']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is True


def test_invalid_year_frequency():
    time_column_name = 'date'
    time_granularity_unit = 'Y'
    time_granularity_step = '2'
    df = pd.DataFrame(
        {time_column_name: [
            '2014-01-01',
            '2016-01-01',
            '2018-01-01',
            '2020-01-01']
        })
    assert check_continuous_time_column(df, time_column_name, time_granularity_unit, time_granularity_step) is False


# TODO unit test for add_future_external_features
def test_add_future_external_features_with_identifiers():
    timeseries_0 = {
        'start': '2018-01-01',
        'target': [12, 13, 14, 15, 16],
        'target_name': 'sales_0',
        'time_column_name': 'date',
        'feat_dynamic_real': [[1, 0, 0, 0, 0], [0, 0, 0, 1, 1]],
        'feat_dynamic_real_columns_names': ['is_holiday', 'is_weekend'],
        'identifiers': {'store': 1, 'item': 1}
    }
    timeseries_1 = {
        'start': '2018-01-01',
        'target': [2, 3, 4, 5, 6],
        'target_name': 'sales_1',
        'time_column_name': 'date',
        'feat_dynamic_real': [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]],
        'feat_dynamic_real_columns_names': ['is_holiday', 'is_weekend'],
        'identifiers': {'store': 1, 'item': 2}
    }
    gluon_train_dataset = ListDataset([timeseries_0, timeseries_1], freq='D')

    external_features_future_df = pd.DataFrame({
        'date': ['2018-01-06', '2018-01-07', '2018-01-08', '2018-01-06', '2018-01-07', '2018-01-08'],
        'store': [1, 1, 1, 1, 1, 1],
        'item': [1, 1, 1, 2, 2, 2],
        'is_holiday': [0, 0, 0, 0, 1, 0],
        'is_weekend': [1, 0, 0, 1, 0, 0]
        })

    gluon_dataset = add_future_external_features(gluon_train_dataset, external_features_future_df)

    assert (gluon_dataset.list_data[0]['feat_dynamic_real'] == np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])).all()
