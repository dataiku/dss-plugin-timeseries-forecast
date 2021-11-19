import pandas as pd
import numpy as np
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from gluonts_forecasts.gluon_dataset import remove_unused_external_features
from gluonts_forecasts.utils import (
    concat_timeseries_per_identifiers,
    concat_all_timeseries,
    add_row_origin,
    quantile_forecasts_series,
)
from dku_constants import (
    METRICS_DATASET,
    METRICS_COLUMNS_DESCRIPTIONS,
    TIMESERIES_KEYS,
    ROW_ORIGIN,
)
from gluonts.model.forecast import QuantileForecast
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


class TrainedModel:
    """
    Wrapper class to make forecasts using a GluonTS Predictor and a training GluonTS ListDataset, and to output a well formatted forecasts dataframe


    Attributes:
        gluon_dataset (gluonts.dataset.common.ListDataset): GluonTS ListDataset generated by the GluonDataset class (with extra fields to name timeseries)
        prediction_length (int): Number of time steps to predict (at most the prediction length used in training)
        quantiles (list): List of forecasts quantiles to compute in the forecasts_df
        time_column_name (str): Time column name used in training
        identifiers_columns (list): List of timeseries identifiers column names used in training.
        forecasts_df (DataFrame): Dataframe with the different quantiles forecasts and the training data if include_history is True
    """

    def __init__(
        self,
        gluon_dataset,
        prediction_length,
        quantiles,
    ):
        self.gluon_dataset = gluon_dataset
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.time_column_name = None
        self.identifiers_columns = None
        self.frequency = gluon_dataset.process.trans[0].freq

    def predict(self, model_label, predictor):
        """
        Use the gluon dataset of training to predict future values and
        concat all forecasts timeseries of different identifiers and quantiles together

        Attributes:
            predictor (gluonts.model.predictor.Predictor)
        """
        self._check(predictor)
        model_name = ModelConfigRegistry().get_model_name_from_label(model_label)
        model_config = ModelConfigRegistry().get_model(model_name)
        if (
            model_name
            and not model_config.can_use_external_feature()
            and TIMESERIES_KEYS.FEAT_DYNAMIC_REAL in self.gluon_dataset.list_data[0]
        ):
            # remove external features from the ListDataset used for predictions if the model cannot use them
            gluon_dataset_without_external_features = remove_unused_external_features(
                self.gluon_dataset, self.frequency
            )
            forecasts = predictor.predict(gluon_dataset_without_external_features)
        else:
            forecasts = predictor.predict(self.gluon_dataset)

        forecasts_list = list(forecasts)

        forecasts_timeseries = self._compute_forecasts_timeseries(forecasts_list)

        multiple_df = concat_timeseries_per_identifiers(forecasts_timeseries)

        forecasts_df = concat_all_timeseries(multiple_df)

        self.time_column_name = self.gluon_dataset.list_data[0][TIMESERIES_KEYS.TIME_COLUMN_NAME]
        self.identifiers_columns = (
            list(self.gluon_dataset.list_data[0][TIMESERIES_KEYS.IDENTIFIERS].keys())
            if TIMESERIES_KEYS.IDENTIFIERS in self.gluon_dataset.list_data[0]
            else []
        )

        if model_name:
            forecasts_df[METRICS_DATASET.MODEL_COLUMN] = model_label

        forecasts_df = forecasts_df.rename(columns={"index": self.time_column_name})

        if TIMESERIES_KEYS.IDENTIFIERS in self.gluon_dataset.list_data[0]:
            forecasts_df = self._reorder_forecasts_df_columns(forecasts_df)

        return forecasts_df

    def _reorder_forecasts_df_columns(self, forecasts_df):
        """Reorder columns with timeseries identifiers columns right after time column"""
        forecasts_columns = [
            column for column in forecasts_df if column not in [self.time_column_name] + self.identifiers_columns
        ]
        return forecasts_df[[self.time_column_name] + self.identifiers_columns + forecasts_columns]

    def format_forecasts_df_for_display(self, forecasts_df, session=None):
        """Format the forecasts dataframe to be displayed to users

        Args:
            session (Timstamp, optional)
        """
        if session:
            forecasts_df[METRICS_DATASET.SESSION] = session

        forecasts_df = forecasts_df.sort_values(
            by=[METRICS_DATASET.MODEL_COLUMN] + self.identifiers_columns + [self.time_column_name],
            ascending=[True] + [True] * len(self.identifiers_columns) + [False],
        )

        return forecasts_df

    def append_history_to_forecasts(self, forecasts_df, history_length_limit=None, session=None):
        """Include the historical data on which the model was trained to the forecasts dataframe.

        Args:
            frequency (str): Used to reconstruct the date range (because a gluon ListDataset only store the start date).
            history_length_limit (int, optional): Maximum number of values to retrieve from historical data per timeseries. Default to None which means all.
            session (Timstamp, optional)

        Returns:
            DataFrame containing both the historical data and the forecasted values.
        """
        history_timeseries = self._retrieve_history_timeseries(self.frequency, history_length_limit)
        multiple_df = concat_timeseries_per_identifiers(history_timeseries)
        history_df = concat_all_timeseries(multiple_df)
        history_df = history_df.rename(columns={"index": self.time_column_name})

        history_forecasts_df = history_df.merge(
            forecasts_df, on=[self.time_column_name] + self.identifiers_columns, how="left", indicator=True
        )
        history_forecasts_df = add_row_origin(
            history_forecasts_df, both=ROW_ORIGIN.FORECAST, left_only=ROW_ORIGIN.HISTORY
        )

        return history_forecasts_df

    def _generate_history_target_series(self, timeseries, frequency, history_length_limit=None):
        """Creates a pandas time series from the past target values with Nan values for the prediction_length future dates.

        Args:
            timeseries (dict): Univariate timeseries dictionary created with the GluonDataset class.
            frequency (str): Used in pandas.date_range.
            history_length_limit (int): Maximum number of values to retrieve from historical data per timeseries. Default to None which means all.

        Returns:
            Series with DatetimeIndex.
        """
        target_series = pd.Series(
            np.append(timeseries[TIMESERIES_KEYS.TARGET], np.repeat(np.nan, self.prediction_length)),
            name=timeseries[TIMESERIES_KEYS.TARGET_NAME],
            index=pd.date_range(
                start=timeseries[TIMESERIES_KEYS.START],
                periods=len(timeseries[TIMESERIES_KEYS.TARGET]) + self.prediction_length,
                freq=frequency,
            ),
        )
        if history_length_limit:
            target_series = target_series.iloc[-(history_length_limit + self.prediction_length) :]
        return target_series

    def _generate_history_external_features_dataframe(self, timeseries, frequency, history_length_limit=None):
        """Creates a pandas time series from the past and future external features values.

        Args:
            timeseries (dict): Univariate timeseries dictionary created with the GluonDataset class.
            frequency (str): Used in pandas.date_range.
            history_length_limit (int): Maximum number of values to retrieve from historical data per timeseries. Default to None which means all.

        Returns:
            DataFrame with DatetimeIndex.
        """
        external_features_df = pd.DataFrame(
            timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL].T[
                : len(timeseries[TIMESERIES_KEYS.TARGET]) + self.prediction_length
            ],
            columns=timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES],
            index=pd.date_range(
                start=timeseries[TIMESERIES_KEYS.START],
                periods=len(timeseries[TIMESERIES_KEYS.TARGET]) + self.prediction_length,
                freq=frequency,
            ),
        )
        if history_length_limit:
            external_features_df = external_features_df.iloc[-(history_length_limit + self.prediction_length) :]
        return external_features_df

    def _retrieve_history_timeseries(self, frequency, history_length_limit=None):
        """Reconstruct the history timeseries from the gluon_dataset object and fill the dates to predict with Nan values.

        Args:
            frequency (str)
            history_length_limit (int): Maximum number of values to retrieve from historical data per timeseries. Default to None which means all.

        Returns:
            Dictionary of list of timeseries by identifiers (None if no identifiers)
        """
        history_timeseries = {}
        for i, timeseries in enumerate(self.gluon_dataset.list_data):
            if TIMESERIES_KEYS.IDENTIFIERS in timeseries:
                timeseries_identifier_key = tuple(sorted(timeseries[TIMESERIES_KEYS.IDENTIFIERS].items()))
            else:
                timeseries_identifier_key = None

            target_series = self._generate_history_target_series(timeseries, frequency, history_length_limit)

            if TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES in timeseries:
                assert (
                    timeseries[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL].shape[1]
                    >= len(timeseries[TIMESERIES_KEYS.TARGET]) + self.prediction_length
                )
                if timeseries_identifier_key not in history_timeseries:
                    external_features_df = self._generate_history_external_features_dataframe(
                        timeseries, frequency, history_length_limit
                    )
                    history_timeseries[timeseries_identifier_key] = [external_features_df]

            if timeseries_identifier_key in history_timeseries:
                history_timeseries[timeseries_identifier_key] += [target_series]
            else:
                history_timeseries[timeseries_identifier_key] = [target_series]
        return history_timeseries

    def _compute_forecasts_timeseries(self, forecasts_list):
        """Compute all forecasts timeseries for each quantile.

        Args:
            forecasts_list (list): List of gluonts.model.forecast.Forecast (objects storing the predicted distributions as samples).

        Returns:
            Dictionary of list of forecasts timeseries by identifiers (None if no identifiers)
        """
        forecasts_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            if TIMESERIES_KEYS.IDENTIFIERS in self.gluon_dataset.list_data[i]:
                timeseries_identifier_key = tuple(
                    sorted(self.gluon_dataset.list_data[i][TIMESERIES_KEYS.IDENTIFIERS].items())
                )
            else:
                timeseries_identifier_key = None

            if i == 0 and isinstance(sample_forecasts, QuantileForecast):
                self.quantiles = self._round_to_existing_quantiles(sample_forecasts)

            for quantile in self.quantiles:
                forecasts_label_prefix = "forecast"
                if quantile < 0.5:
                    forecasts_label_prefix += "_lower"
                elif quantile > 0.5:
                    forecasts_label_prefix += "_upper"

                forecasts_series = (
                    quantile_forecasts_series(sample_forecasts, quantile, self.frequency)
                    .rename(f"{forecasts_label_prefix}_{self.gluon_dataset.list_data[i][TIMESERIES_KEYS.TARGET_NAME]}")
                    .iloc[: self.prediction_length]
                )
                if timeseries_identifier_key in forecasts_timeseries:
                    forecasts_timeseries[timeseries_identifier_key] += [forecasts_series]
                else:
                    forecasts_timeseries[timeseries_identifier_key] = [forecasts_series]
        return forecasts_timeseries

    def create_forecasts_column_description(self, forecasts_df):
        """Explain the meaning of the forecasts columns"""
        column_descriptions = METRICS_COLUMNS_DESCRIPTIONS
        confidence_interval = self._retrieve_confidence_interval()
        for column in forecasts_df.columns:
            if "forecast_lower_" in column:
                column_descriptions[
                    column
                ] = f"Lower bound of the {confidence_interval}% forecasts confidence interval."
            elif "forecast_upper_" in column:
                column_descriptions[
                    column
                ] = f"Upper bound of the {confidence_interval}% forecasts confidence interval."
            elif "forecast_" in column:
                column_descriptions[column] = "Median of probabilistic forecasts"
        return column_descriptions

    def _check(self, predictor):
        """Raises ValueError if the selected prediction_length is higher than the one used in training"""
        if predictor.prediction_length < self.prediction_length:
            raise ValueError(
                f"Please choose a forecasting horizon lower or equal to the one chosen when training: {predictor.prediction_length}"
            )

    def _round_to_existing_quantiles(self, sample_forecasts):
        """Find the quantiles that exists in sample_forecasts that are closest to the selected quantiles.
        QuantileForecast cannot predict all quantiles but only a list predifined during training.

        Args:
            sample_forecasts (QuantileForecast)

        Returns:
            List of quantiles that exists in the sample_forecasts
        """
        new_quantiles = []
        possible_quantiles = list(map(float, sample_forecasts.forecast_keys))
        for quantile in self.quantiles:
            new_quantiles += [min(possible_quantiles, key=lambda x: abs(x - quantile))]
        return new_quantiles

    def _retrieve_confidence_interval(self):
        """Retrieve the confidence interval percentage from the minimum and maximum quantiles.
        If they are not symetric around 0.5, log a warning.

        Returns:
            Integer representing the percentage of the confidence interval.
        """
        lower_quantile, upper_quantile = min(self.quantiles), max(self.quantiles)
        confidence_interval = round((upper_quantile - lower_quantile) * 100)
        if round((1 - upper_quantile) * 100, 2) != round(lower_quantile * 100, 2):
            logger.warning(
                f"The output confidence interval is not centered around the median. Lower and upper quantiles are [{lower_quantile}, {upper_quantile}]"
            )
        return confidence_interval
