import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.tseries.offsets import Tick, BusinessDay, Week, MonthEnd
from pandas.tseries.frequencies import to_offset
from timeseries_preparation.h2g2 import H2G2
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


class TimeseriesPreparator:
    """
    Class to check the timeseries has the right data and prepare it to have regular date interval

    Attributes:
        time_column_name (str): Name of the time column
        frequency (list): Pandas timeseries frequency (e.g. '3M')
        target_columns_names (list, optional): List of column names to predict
        timeseries_identifiers_names (list, optional): Columns to identify multiple time series when data is in long format
        external_features_columns_names (list, optional): List of columns with dynamic real features over time
        max_timeseries_length (int, optional): Maximum number of records to keep in the timeseries
        timeseries_identifiers_values (list, optional): List of dict with timeseries identifiers name as keys and timeseries identifiers values as values
        prediction_length (int, optional): Number of records to predict
    """

    def __init__(
        self,
        time_column_name,
        frequency,
        target_columns_names=None,
        timeseries_identifiers_names=None,
        external_features_columns_names=None,
        max_timeseries_length=None,
        timeseries_identifiers_values=None,
        prediction_length=None,
    ):
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.target_columns_names = target_columns_names
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.external_features_columns_names = external_features_columns_names
        self.max_timeseries_length = max_timeseries_length
        self.timeseries_identifiers_values = timeseries_identifiers_values
        self.prediction_length = prediction_length

    def prepare_timeseries_dataframe(self, dataframe):
        """Convert time column to pandas.Datetime without timezones. Truncate dates to selected frequency.
        Check that there are no duplicate dates and that there are no missing dates.
        Sort timeseries. Keep only the most recent dates of each timeseries if specified.

        Args:
            dataframe (DataFrame)

        Raises:
            ValueError: If the time column cannot be parsed as a date by pandas.

        Returns:
            Prepared timeseries
        """
        self._check_data(dataframe)

        dataframe_prepared = dataframe.copy()

        try:
            dataframe_prepared[self.time_column_name] = pd.to_datetime(dataframe[self.time_column_name]).dt.tz_localize(
                tz=None
            )
        except Exception:
            raise ValueError(f"Please parse the date column '{self.time_column_name}' in a Prepare recipe")

        dataframe_prepared = self._truncate_dates(dataframe_prepared)

        dataframe_prepared = self._sort(dataframe_prepared)

        self._check_regular_frequency(dataframe_prepared)
        log_message_prefix = "Found"
        self._log_timeseries_lengths(dataframe_prepared, log_message_prefix=log_message_prefix)

        if self.max_timeseries_length:
            dataframe_prepared = self._keep_last_dates(dataframe_prepared)
            log_message_prefix = f"Sampling {self.max_timeseries_length} last records, obtained"
            self._log_timeseries_lengths(dataframe_prepared, log_message_prefix=log_message_prefix)

        if self.timeseries_identifiers_names:
            if self.timeseries_identifiers_values:
                self._check_identifiers_values(dataframe_prepared)
            else:
                self.timeseries_identifiers_values = (
                    dataframe_prepared[self.timeseries_identifiers_names].drop_duplicates().to_dict("records")
                )

        return dataframe_prepared

    def _check_identifiers_values(self, dataframe):
        # raise Exception if not identical values for each timeseries identifiers
        # compare self.timeseries_identifiers_values and dataframe[self.timeseries_identifiers_names].drop_duplicates().to_dict("records")
        # TODO
        pass

    def check_schema_from_dataset(self, dataset_schema):
        # TODO
        pass

    def _check_data(self, df):
        self._check_not_empty_dataframe(df)
        self._check_timeseries_identifiers_columns_types(df)
        self._check_no_missing_values(df)

    def _truncate_dates(self, df):
        """Truncate dates to selected frequency. For Week/Month/Year, truncate to end of Week/Month/Year.
        Check there are no duplicate dates.

        Examples:
            '2020-12-15 12:45:30' becomes '2020-12-15 12:40:00' with frequency '20min'
            '2020-12-15 12:00:00' becomes '2020-12-15 00:00:00' with frequency '24H'
            '2020-12-15 12:30:00' becomes '2020-12-15 00:00:00' with frequency 'D'
            '2020-12-15 12:30:00' becomes '2020-12-31 00:00:00' with frequency 'M'
            '2020-12-15 12:30:00' becomes '2021-12-31 00:00:00' with frequency '6M'

        Args:
            df (DataFrame): Dataframe in wide or long format with a time column.

        Raises:
            ValueError: If there are duplicates dates before or after truncation.

        Returns:
            Sorted DataFrame with truncated dates.
        """
        df_truncated = df.copy()

        error_message_suffix = (
            ". Please check the Long format parameter." if not self.timeseries_identifiers_names else "."
        )
        self._check_duplicate_dates(df_truncated, error_message_suffix=error_message_suffix)

        frequency_offset = to_offset(self.frequency)
        if isinstance(frequency_offset, Tick):
            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor(self.frequency)
        elif isinstance(frequency_offset, BusinessDay):
            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor("D")
        else:
            if isinstance(frequency_offset, Week):
                truncation_offset = pd.offsets.Week(weekday=frequency_offset.weekday, n=0)
            elif isinstance(frequency_offset, MonthEnd):
                truncation_offset = pd.offsets.MonthEnd(n=0)

            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor("D") + truncation_offset

        self._log_truncation(df_truncated, df)

        error_message_suffix = (
            f" after truncation to '{self.frequency}' frequency. Please check the Frequency parameter."
        )
        self._check_duplicate_dates(df_truncated, error_message_suffix=error_message_suffix)

        return df_truncated

    def _sort(self, df):
        """Return a DataFrame sorted by timeseries identifiers and time column (both ascending)"""
        return df.sort_values(by=(self.timeseries_identifiers_names or []) + [self.time_column_name])

    def _check_regular_frequency(self, df):
        """Check that time column exactly equals the pandas.dat_range with selected frequency"""
        if self.timeseries_identifiers_names:
            for _, identifiers_df in df.groupby(self.timeseries_identifiers_names):
                assert_time_column_valid(identifiers_df, self.time_column_name, self.frequency)
        else:
            assert_time_column_valid(df, self.time_column_name, self.frequency)

    def _keep_last_dates(self, df):
        """Keep only at most the last max_timeseries_length dates of each timeseries.

        Args:
            df (DataFrame)

        Returns:
            Filtered dataframe
        """
        if self.max_timeseries_length == 42:
            print(H2G2)
        if not (self.timeseries_identifiers_names or []):
            return df.tail(self.max_timeseries_length)
        else:
            return (
                df.groupby((self.timeseries_identifiers_names or []))
                .apply(lambda x: x.tail(self.max_timeseries_length))
                .reset_index(drop=True)
            )

    def _log_truncation(self, df_truncated, df):
        """Log how many dates were truncated for users to understand how their data were changed

        Args:
            df_truncated (DataFrame): Dataframe after truncation
            df (DataFrame): Original dataframe

        """
        total_dates = len(df_truncated.index)
        truncated_dates = (df_truncated[self.time_column_name] != df[self.time_column_name]).sum()
        if truncated_dates > 0:
            logger.warning(
                f"Dates truncated to {frequency_custom_label(self.frequency)} frequency: {total_dates - truncated_dates} dates kept, {truncated_dates} dates truncated"
            )
            if truncated_dates == total_dates:
                self._check_end_of_week_frequency(df_truncated, df)
        else:
            logger.info(f"No dates were changed after truncation to {frequency_custom_label(self.frequency)} frequency")

    def _check_end_of_week_frequency(self, df_truncated, df):
        """Check not all that truncated days are different days"""
        frequency_offset = to_offset(self.frequency)
        if isinstance(frequency_offset, Week):
            if all(df_truncated[self.time_column_name].dt.dayofweek != df[self.time_column_name].dt.dayofweek):
                raise ValueError(
                    f"No weekly dates on {WEEKDAYS[frequency_offset.weekday]}. Please check the 'End of week day' parameter."
                )

    def _check_duplicate_dates(self, df, error_message_suffix=None):
        """Check dataframe has no duplicate dates and raise an actionable error message"""
        duplicate_dates = self._count_duplicate_dates(df)
        if duplicate_dates > 0:
            error_message = f"Input dataset has {duplicate_dates} duplicate dates"
            if error_message_suffix:
                error_message += error_message_suffix
            raise ValueError(error_message)

    def _count_duplicate_dates(self, df):
        """Return total number of duplicates dates within all timeseries"""
        return df.duplicated(
            subset=(self.timeseries_identifiers_names or []) + [self.time_column_name], keep=False
        ).sum()

    def _check_not_empty_dataframe(self, df):
        if len(df.index) == 0:
            raise ValueError(f"Input dataframe is empty")

    def _check_timeseries_identifiers_columns_types(self, df):
        """Raises ValueError if a timeseries identifiers column is not numerical or string"""
        if self.timeseries_identifiers_names:
            invalid_columns = []
            for column_name in self.timeseries_identifiers_names:
                if not is_numeric_dtype(df[column_name]) and not is_string_dtype(df[column_name]):
                    invalid_columns += [column_name]
            if len(invalid_columns) > 0:
                raise ValueError(
                    f"Time series identifiers columns '{invalid_columns}' must be of string or numeric type. Please change the type in a Prepare recipe."
                )

    def _check_no_missing_values(self, df):
        invalid_columns = []
        for column_name in (
            [self.time_column_name]
            + (self.target_columns_names or [])
            + (self.timeseries_identifiers_names or [])
            + (self.external_features_columns_names or [])
        ):
            if df[column_name].isnull().values.any():
                invalid_columns += [column_name]
        if len(invalid_columns) > 0:
            raise ValueError(
                f"Column(s) '{invalid_columns}' have missing values. You can use the Time Series Preparation plugin to resample your time series."
            )

    def _log_timeseries_lengths(self, df, log_message_prefix=None):
        """Log the number and sizes of time series and whether it's after sampling or not"""
        if not self.timeseries_identifiers_names:
            timeseries_lengths = [len(df.index)]
        else:
            timeseries_lengths = list(df.groupby(self.timeseries_identifiers_names).size())
        log_message = f"{len(timeseries_lengths)} time series"
        if all(length == timeseries_lengths[0] for length in timeseries_lengths):
            log_message += f" of {timeseries_lengths[0]} records"
        else:
            log_message += f" of {min(timeseries_lengths)} to {max(timeseries_lengths)} records"
        if log_message_prefix:
            logger.info(f"{log_message_prefix} {log_message}")


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
        error_message = f"Time column '{time_column_name}' has missing values with frequency '{frequency}'."
        error_message += " You can use the Time Series Preparation plugin to resample your time column."
        raise ValueError(error_message)


FREQUENCY_LABEL = {"T": "minute", "H": "hour", "D": "day", "B": "business day"}


WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def frequency_custom_label(frequency):
    frequency_offset = to_offset(frequency)
    if isinstance(frequency_offset, MonthEnd):
        if frequency_offset.n == 3:
            return "quarter"
        elif frequency_offset.n == 6:
            return "semester"
        elif frequency_offset.n == 12:
            return "year"
        elif frequency_offset.n == 1:
            return "end of month"
        else:
            return f"{frequency_offset.n} months"
    elif isinstance(frequency_offset, Week):
        prefix = f"{frequency_offset.n} weeks" if frequency_offset.n > 1 else "week"
        return f"{prefix} ending on {WEEKDAYS[frequency_offset.weekday]}"
    else:
        prefix = f"{frequency_offset.n} " if frequency_offset.n > 1 else ""
        middle = f"{FREQUENCY_LABEL[frequency_offset.name]}"
        suffix = "s" if frequency_offset.n > 1 else ""
        return f"{prefix}{middle}{suffix}"
