import pandas as pd
import numpy as np
from functools import reduce
import copy
from constants import TIMESERIES_KEYS


def apply_filter_conditions(df, conditions):
    """Apply filter conditions on df.

    Args:
        df (DataFrame): Dataframe to filter.
        conditions (list): List of DataFrame filtering conditions on df.

    Returns:
        Filtered DataFrame.
    """
    if len(conditions) == 0:
        return df
    elif len(conditions) == 1:
        return df[conditions[0]]
    else:
        return df[reduce(lambda c1, c2: c1 & c2, conditions[1:], conditions[0])]


def add_future_external_features(gluon_train_dataset, external_features_future_df, prediction_length):
    """Append the future external features to the 'feat_dynamic_real' arrays of each timeseries of the ListDataset used for training.

    Args:
        gluon_train_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the GluonDataset class.
        external_features_future_df (DataFrame): Dataframe of future (dated after timeseries of gluon_train_dataset) external features.
        prediction_length (int): To check that external_features_future_df has the right length.

    Raises:
        ValueError: If the length of external_features_future_df is not prediction_length.

    Returns:
        gluonts.dataset.common.ListDataset with future external features.
    """
    gluon_dataset = copy.deepcopy(gluon_train_dataset)
    for i, timeseries in enumerate(gluon_train_dataset):
        if TIMESERIES_KEYS.IDENTIFIERS in timeseries:
            timeseries_identifiers = timeseries[TIMESERIES_KEYS.IDENTIFIERS]
            conditions = [external_features_future_df[k] == v for k, v in timeseries_identifiers.items()]
            timeseries_external_features_future_df = apply_filter_conditions(external_features_future_df, conditions)
        else:
            timeseries_external_features_future_df = external_features_future_df

        feat_dynamic_real_train = timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL]
        feat_dynamic_real_columns_names = timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES]

        feat_dynamic_real_future = timeseries_external_features_future_df[feat_dynamic_real_columns_names].values.T

        if feat_dynamic_real_future.shape[1] != prediction_length:
            raise ValueError("Length of future external features timeseries must be equal to the training prediction length ({})".format(prediction_length))

        feat_dynamic_real_appended = np.append(feat_dynamic_real_train, feat_dynamic_real_future, axis=1)

        gluon_dataset.list_data[i][TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = feat_dynamic_real_appended

    return gluon_dataset


def assert_time_column_valid(dataframe, time_column_name, frequency, start_date=None, periods=None):
    """Assert that the time column has the same values as the pandas.date_range generated with frequency and the first and last row of dataframe[time_column_name] 
    (or with start_date and periods if specified).

    Args:
        dataframe (DataFrame)
        time_column_name (str)
        frequency (str): Use as frequency of pandas.date_range.
        start_date (pandas.Timestamp, optional): Use as start_date of pandas.date_range if specified. Defaults to None.
        periods (int, optional): Use as periods of pandas.date_range if specified. Defaults to None.

    Raises:
        ValueError: If the time column doesn't have regular time intervals of the chosen frequency.
    """
    if start_date is None:
        start_date = dataframe[time_column_name].iloc[0]
    if periods:
        date_range_values = pd.date_range(start=start_date, periods=periods, freq=frequency).values
    else:
        end_date = dataframe[time_column_name].iloc[-1]
        date_range_values = pd.date_range(start=start_date, end=end_date, freq=frequency).values

    if not np.array_equal(dataframe[time_column_name].values, date_range_values):
        error_message = "Time column {} doesn't have regular time intervals of frequency {}.".format(time_column_name, frequency)
        if frequency.endswith(("M", "Y")):
            error_message += " For Month (or Year) frequency, timestamps must be end of Month (or Year) (for e.g. '2020-12-31 00:00:00')."
        error_message += " You should use the timeseries preparation plugin first to resample the date column."
        raise ValueError(error_message)


def concat_timeseries_per_identifiers(all_timeseries):
    """Concatenate on columns all forecasts timeseries with same identifiers.

    Args:
        all_timeseries (dict): Dictionary of timeseries (value) by identifiers dictionary (key).

    Returns:
        List of timeseries with multiple forecasts for each identifiers.
    """
    multiple_df = []
    for timeseries_identifier_key, series_list in all_timeseries.items():
        unique_identifiers_df = pd.concat(series_list, axis=1).reset_index(drop=False)
        if timeseries_identifier_key:
            for identifier_key, identifier_value in timeseries_identifier_key:
                unique_identifiers_df[identifier_key] = identifier_value
        multiple_df += [unique_identifiers_df]
    return multiple_df


def concat_all_timeseries(multiple_df):
    """Concatenate on rows all forecasts timeseries (one identifiers timeseries after the other).

    Args:
        multiple_df (list): List of multivariate timeseries DataFrame with identifiers columns.

    Returns:
        DataFrame of multivariate long format timeseries.
    """
    return pd.concat(multiple_df, axis=0).reset_index(drop=True)
