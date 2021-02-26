from gluonts.dataset.common import ListDataset
from dku_constants import TIMESERIES_KEYS
import numpy as np


class GluonDataset:
    """
    Wrapper class to generate a GluonTS ListDataset with multiple time series based on the target and identifiers columns
    Each timeseries stores information about its target(s), time and external features column names as well as its identifiers values

    Attributes:
        dataframe (DataFrame)
        time_column_name (list)
        frequency (str): Pandas timeseries frequency (e.g. '3M')
        target_columns_names (list): List of column names to predict
        timeseries_identifiers_names (list): Columns to identify multiple time series when data is in long format
        external_features_columns_names (list): List of columns with dynamic real features over time
    """

    def __init__(
        self,
        dataframe,
        time_column_name,
        frequency,
        target_columns_names,
        timeseries_identifiers_names=None,
        external_features_columns_names=None,
        min_length=None,
    ):
        self.dataframe = dataframe
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.target_columns_names = target_columns_names
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.external_features_columns_names = external_features_columns_names
        self.min_length = min_length

    def create_list_datasets(self, cut_lengths=[]):
        """Create timeseries for each identifier tuple and each target.

        Args:
            cut_length (int, optional): Remove the last cut_length time steps of each timeseries. Defaults to empty list.

        Returns:
            List of gluonts.dataset.common.ListDataset with extra keys for each timeseries
        """
        multivariate_timeseries_per_cut_length = [[] for cut_length in cut_lengths]
        if self.timeseries_identifiers_names:
            for identifiers_values, identifiers_df in self.dataframe.groupby(self.timeseries_identifiers_names):
                for cut_length_index, cut_length in enumerate(cut_lengths):
                    multivariate_timeseries_per_cut_length[cut_length_index] += self._create_gluon_multivariate_timeseries(
                        identifiers_df, cut_length, identifiers_values=identifiers_values
                    )
        else:
            for cut_length_index, cut_length in enumerate(cut_lengths):
                multivariate_timeseries_per_cut_length[cut_length_index] += self._create_gluon_multivariate_timeseries(self.dataframe, cut_length)
        gluon_list_dataset_per_cut_length = []
        for multivariate_timeseries in multivariate_timeseries_per_cut_length:
            gluon_list_dataset_per_cut_length += [ListDataset(multivariate_timeseries, freq=self.frequency)]
        return gluon_list_dataset_per_cut_length

    def _create_gluon_multivariate_timeseries(self, dataframe, cut_length, identifiers_values=None):
        """Create a list of timeseries dictionaries for each target column

        Args:
            dataframe (DataFrame): Timeseries dataframe with one or multiple target columns.
            cut_length (int): Remove the last cut_length time steps of each timeseries.
            identifiers_values (obj/tuple, optional): Values or tuple of values of the groupby. Defaults to None.

        Returns:
            List of timeseries dictionaries
        """
        self._check_minimum_length(dataframe, cut_length)
        multivariate_timeseries = []
        for target_column_name in self.target_columns_names:
            multivariate_timeseries.append(self._create_gluon_univariate_timeseries(dataframe, target_column_name, cut_length, identifiers_values))
        return multivariate_timeseries

    def _create_gluon_univariate_timeseries(self, dataframe, target_column_name, cut_length, identifiers_values=None):
        """Create a dictionary with keys to store information about the timeseries:
            - start (Pandas.Timestamp): start date of timeseries
            - target (numpy.array): array of target values
            - target_name (str): name of target column
            - time_column_name (str)
            - feat_dynamic_real (numpy.array, optional): array of external features of shape features x values
            - feat_dynamic_real_columns_names (list, optional): list of external features column names
            - identifiers (dict, optional): dictionary of identifiers values (value) by identifiers column name (key)

        Args:
            dataframe (DataFrame): Timeseries dataframe with one or multiple target columns
            target_column_name (str)
            cut_length (int): Remove the last cut_length time steps of each timeseries.
            identifiers_values (obj/tuple, optional): Values or tuple of values of the groupby. Defaults to None.

        Returns:
            Dictionary for one timeseries
        """
        length = -cut_length if cut_length > 0 else None
        univariate_timeseries = {
            TIMESERIES_KEYS.START: dataframe[self.time_column_name].iloc[0],
            TIMESERIES_KEYS.TARGET: dataframe[target_column_name].iloc[:length].values,
            TIMESERIES_KEYS.TARGET_NAME: target_column_name,
            TIMESERIES_KEYS.TIME_COLUMN_NAME: self.time_column_name,
        }
        if self.external_features_columns_names:
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = dataframe[self.external_features_columns_names].iloc[:length].values.T
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES] = self.external_features_columns_names
        if identifiers_values is not None:
            if len(self.timeseries_identifiers_names) > 1:
                identifiers_map = {self.timeseries_identifiers_names[i]: identifier_value for i, identifier_value in enumerate(identifiers_values)}
            else:
                identifiers_map = {self.timeseries_identifiers_names[0]: identifiers_values}
            univariate_timeseries[TIMESERIES_KEYS.IDENTIFIERS] = identifiers_map
        return univariate_timeseries

    def _check_minimum_length(self, dataframe, cut_length):
        """Check that the timeseries dataframe has enough values.

        Args:
            dataframe (DataFrame): Timeseries dataframe with one or multiple target columns
            cut_length (int): Numnber of time steps that will be removed from each timeseries.

        Raises:
            ValueError: If the dataframe doesn't have enough values.
        """
        min_length = self.min_length
        if cut_length:
            min_length += cut_length
        if len(dataframe.index) < min_length:
            raise ValueError(f"Time series must have at least {min_length} values")


def remove_unused_external_features(list_dataset, frequency):
    """Copy a gluon list dataset and remove the external features fields

    Args:
        list_dataset (ListDataset): Gluon ListDataset with FEAT_DYNAMIC_REAL fields
        frequency (str)

    Returns:
        A ListDataset without FEAT_DYNAMIC_REAL fields
    """    
    new_list_dataset = []
    for data in list_dataset.list_data:
        new_data = data.copy()
        new_data.pop(TIMESERIES_KEYS.FEAT_DYNAMIC_REAL, None)
        new_data.pop(TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES, None)
        new_list_dataset.append(new_data)
    return ListDataset(new_list_dataset, freq=frequency)
