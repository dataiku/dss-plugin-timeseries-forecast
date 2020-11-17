from gluonts_forecasts.utils import assert_time_column_valid
from gluonts.dataset.common import ListDataset
from constants import TIMESERIES_KEYS


class GluonDataset:
    """
    Wrapper class to generate a GluonTS ListDataset with multiple time series based on the target and identifiers columns
    Each timeseries stores information about its target(s), time and external features column names as well as its identifiers values

    Attributes:
        dataframe (Pandas.DataFrame)
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
    ):
        self.dataframe = dataframe
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.target_columns_names = target_columns_names
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.external_features_columns_names = external_features_columns_names

    def create_list_dataset(self, cut_length=None):
        """
        return a GluonTS ListDataset of timeseries for each identifier tuple and each target
        and remove the last cut_length time steps of each timeseries
        """
        length = -cut_length if cut_length else None
        multivariate_timeseries = []
        if self.timeseries_identifiers_names:
            start_date, periods = None, None
            for i, (identifiers_values, identifiers_df) in enumerate(self.dataframe.groupby(self.timeseries_identifiers_names)):
                assert_time_column_valid(
                    identifiers_df,
                    self.time_column_name,
                    self.frequency,
                    start_date=start_date,
                    periods=periods,
                )
                if i == 0:
                    start_date = identifiers_df[self.time_column_name].iloc[0]
                    periods = len(identifiers_df.index)
                multivariate_timeseries += self._create_gluon_multivariate_timeseries(identifiers_df, length, identifiers_values=identifiers_values)
        else:
            assert_time_column_valid(self.dataframe, self.time_column_name, self.frequency)
            multivariate_timeseries += self._create_gluon_multivariate_timeseries(self.dataframe, length)
        return ListDataset(multivariate_timeseries, freq=self.frequency)

    def _create_gluon_multivariate_timeseries(self, df, length, identifiers_values=None):
        """ return a list of timeseries dictionaries for each target column """
        multivariate_timeseries = []
        for target_column_name in self.target_columns_names:
            multivariate_timeseries.append(self._create_gluon_univariate_timeseries(df, target_column_name, length, identifiers_values))
        return multivariate_timeseries

    def _create_gluon_univariate_timeseries(self, df, target_column_name, length, identifiers_values=None):
        """ return a dictionary for one timeseries and add extra features and identifiers columns if specified """
        univariate_timeseries = {
            TIMESERIES_KEYS.START: df[self.time_column_name].iloc[0],
            TIMESERIES_KEYS.TARGET: df[target_column_name].iloc[:length].values,
            TIMESERIES_KEYS.TARGET_NAME: target_column_name,
            TIMESERIES_KEYS.TIME_COLUMN_NAME: self.time_column_name,
        }
        if self.external_features_columns_names:
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = df[self.external_features_columns_names].iloc[:length].values.T
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES] = self.external_features_columns_names
        if identifiers_values:
            if len(self.timeseries_identifiers_names) > 1:
                identifiers_map = {self.timeseries_identifiers_names[i]: identifier_value for i, identifier_value in enumerate(identifiers_values)}
            else:
                identifiers_map = {self.timeseries_identifiers_names[0]: identifiers_values}
            univariate_timeseries[TIMESERIES_KEYS.IDENTIFIERS] = identifiers_map
        return univariate_timeseries
