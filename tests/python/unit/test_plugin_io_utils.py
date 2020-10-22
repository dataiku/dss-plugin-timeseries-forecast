from plugin_io_utils import dummy_test_function
import pandas as pd

#dataset = {'time_column': ['', '', ''], 'data_column': [0, 1, 2]}
dataset_timeline_gap = {'time_column': ['2013-02-08T00:00:00.000Z', '2013-02-11T00:00:00.000Z', '2013-02-12T00:00:00.000Z'], 'data_column': [0, 1, 2]}
dataset_timeline = {'time_column': ['2013-02-10T00:00:00.000Z', '2013-02-11T00:00:00.000Z', '2013-02-12T00:00:00.000Z'], 'data_column': [0, 1, 2]}

dataframe_timeline = pd.DataFrame.from_dict(dataset_timeline)
dataframe_timeline['time_column'] = pd.to_datetime(dataframe_timeline['time_column'])
dataframe_timeline_gap = pd.DataFrame.from_dict(dataset_timeline_gap)
dataframe_timeline_gap['time_column'] = pd.to_datetime(dataframe_timeline_gap['time_column'])

def test_no_function():
    print("ALX:{}".format(dataframe_timeline))

# def test_is_timeline_continuous():
#     assert is_timeline_continuous(dataframe_timeline, 'time_column') == True
    
# def test_is_timeline_continuous_gap():
#     assert is_timeline_continuous(dataframe_timeline_gap, 'time_column') == False

def test_dummy_test_function():
    assert dummy_test_function() == True
