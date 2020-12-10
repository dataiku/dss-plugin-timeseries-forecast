from gluonts_forecasts.utils import assert_time_column_valid
from gluonts.dataset.common import ListDataset
from constants import TIMESERIES_KEYS


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

    def create_list_dataset(self, cut_length=None):
        """Create timeseries for each identifier tuple and each target.
        Check that the time column is valid (regular time steps of the chosen frequency and they all have the same start date).

        Args:
            cut_length (int, optional): Remove the last cut_length time steps of each timeseries. Defaults to None.

        Returns:
            gluonts.dataset.common.ListDataset with extra keys for each timeseries
        """
        multivariate_timeseries = []
        if self.timeseries_identifiers_names:
            start_date, periods = None, None
            for i, (identifiers_values, identifiers_df) in enumerate(self.dataframe.groupby(self.timeseries_identifiers_names)):
                identifiers_df = identifiers_df.sort_values(by=self.time_column_name, ascending=True)
                assert_time_column_valid(
                    identifiers_df,
                    self.time_column_name,
                    self.frequency,
                    start_date=start_date,
                    periods=periods,
                )
                # if i == 0:
                #     start_date = identifiers_df[self.time_column_name].iloc[0]
                #     periods = len(identifiers_df.index)
                multivariate_timeseries += self._create_gluon_multivariate_timeseries(identifiers_df, cut_length, identifiers_values=identifiers_values)
        else:
            self.dataframe = self.dataframe.sort_values(by=self.time_column_name, ascending=True)
            assert_time_column_valid(self.dataframe, self.time_column_name, self.frequency)
            multivariate_timeseries += self._create_gluon_multivariate_timeseries(self.dataframe, cut_length)
        return ListDataset(multivariate_timeseries, freq=self.frequency)

    def _create_gluon_multivariate_timeseries(self, dataframe, cut_length, identifiers_values=None):
        """Create a list of timeseries dictionaries for each target column

        Args:
            dataframe (DataFrame): Timeseries dataframe with one or multiple target columns.
            cut_length (int): Remove the last cut_length time steps of each timeseries. Can be None.
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
            cut_length (int): Remove the last cut_length time steps of each timeseries. Can be None.
            identifiers_values (obj/tuple, optional): Values or tuple of values of the groupby. Defaults to None.

        Returns:
            Dictionary for one timeseries
        """
        length = -cut_length if cut_length else None
        univariate_timeseries = {
            TIMESERIES_KEYS.START: dataframe[self.time_column_name].iloc[0],
            TIMESERIES_KEYS.TARGET: dataframe[target_column_name].iloc[:length].values,
            TIMESERIES_KEYS.TARGET_NAME: target_column_name,
            TIMESERIES_KEYS.TIME_COLUMN_NAME: self.time_column_name,
        }
        if self.external_features_columns_names:
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL] = dataframe[self.external_features_columns_names].iloc[:length].values.T
            univariate_timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES] = self.external_features_columns_names
        if identifiers_values:
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
            cut_length (int): Numnber of time steps that will be removed from each timeseries. Can be None.

        Raises:
            ValueError: If the dataframe doesn't have enough values.
        """
        min_length = self.min_length
        if cut_length:
            min_length += cut_length
        if len(dataframe.index) < min_length:
            raise ValueError("Time series must have at least {} values".format(min_length))
