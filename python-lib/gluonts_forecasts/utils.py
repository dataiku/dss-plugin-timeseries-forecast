import pandas as pd
import numpy as np
from functools import reduce
import copy


def apply_filter_conditions(df, conditions):
    """
    return a function to apply filtering conditions on df
    """
    if len(conditions) == 0:
        return df
    elif len(conditions) == 1:
        return df[conditions[0]]
    else:
        return df[reduce(lambda c1, c2: c1 & c2, conditions[1:], conditions[0])]


def add_future_external_features(gluon_train_dataset, external_features_future_df, prediction_length):
    """ append the future external features to the gluonTS ListDataset used for training """
    gluon_dataset = copy.deepcopy(gluon_train_dataset)
    for i, timeseries in enumerate(gluon_train_dataset):
        if "identifiers" in timeseries:
            timeseries_identifiers = timeseries["identifiers"]
            conditions = [external_features_future_df[k] == v for k, v in timeseries_identifiers.items()]
            timeseries_external_features_future_df = apply_filter_conditions(external_features_future_df, conditions)
        else:
            timeseries_external_features_future_df = external_features_future_df

        feat_dynamic_real_train = timeseries["feat_dynamic_real"]
        feat_dynamic_real_columns_names = timeseries["feat_dynamic_real_columns_names"]

        feat_dynamic_real_future = timeseries_external_features_future_df[feat_dynamic_real_columns_names].values.T

        if feat_dynamic_real_future.shape[1] != prediction_length:
            raise ValueError("Length of future external features timeseries must be equal to the training prediction length ({})".format(prediction_length))

        feat_dynamic_real_appended = np.append(feat_dynamic_real_train, feat_dynamic_real_future, axis=1)

        gluon_dataset.list_data[i]["feat_dynamic_real"] = feat_dynamic_real_appended

    return gluon_dataset


def assert_time_column_valid(dataframe, time_column_name, frequency, start_date=None, periods=None):
    if not start_date:
        start_date = dataframe[time_column_name].iloc[0]
    if periods:
        date_range_values = pd.date_range(start=start_date, periods=periods, freq=frequency).values
    else:
        end_date = dataframe[time_column_name].iloc[-1]
        date_range_values = pd.date_range(start=start_date, end=end_date, freq=frequency).values

    if not np.array_equal(dataframe[time_column_name].values, date_range_values):
        error_message = "Time column {} doesn't have regular time intervals of frequency {}.".format(time_column_name, frequency)
        if frequency.endswith(("M", "Y")):
            error_message += "For Month (or Year) frequency, timestamps must be end of Month (or Year) (for e.g. '2020-12-31 00:00:00')"
        raise ValueError(error_message)
