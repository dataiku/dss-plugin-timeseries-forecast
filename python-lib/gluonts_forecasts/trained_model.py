import pandas as pd
import numpy as np


class TrainedModel:
    def __init__(self, predictor, gluon_dataset, prediction_length, quantiles, include_history):
        self.predictor = predictor
        self.gluon_dataset = gluon_dataset
        self.prediction_length = predictor.prediction_length if prediction_length == 0 else prediction_length
        self.quantiles = quantiles
        self.include_history = include_history
        self.forecasts_df = None
        self._check()

    def predict(self):
        """
        use the gluon dataset of training to predict future values and
        concat all forecasts timeseries of different identifiers and quantiles together
        """
        forecasts = self.predictor.predict(self.gluon_dataset)
        forecasts_list = list(forecasts)

        forecasts_timeseries = self._compute_forecasts_timeseries(forecasts_list)

        multiple_df = self._concat_timeseries_per_identifiers(forecasts_timeseries)

        self.forecasts_df = self._concat_all_timeseries(multiple_df)

        time_column_name = self.gluon_dataset.list_data[0]["time_column_name"]
        identifiers_columns = list(self.gluon_dataset.list_data[0]["identifiers"].keys()) if "identifiers" in self.gluon_dataset.list_data[0] else []

        if self.include_history:
            frequency = forecasts_list[0].freq
            self.forecasts_df = self._include_history(frequency, identifiers_columns)

        self.forecasts_df = self.forecasts_df.rename(columns={"index": time_column_name})

        if "identifiers" in self.gluon_dataset.list_data[0]:
            self._reorder_forecasts_df(time_column_name, identifiers_columns)

    def _include_history(self, frequency, identifiers_columns):
        history_timeseries = self._retrieve_history_timeseries(frequency)
        multiple_df = self._concat_timeseries_per_identifiers(history_timeseries)
        history_df = self._concat_all_timeseries(multiple_df)
        return history_df.merge(self.forecasts_df, on=["index"] + identifiers_columns, how="left")

    def _generate_history_target_series(self, timeseries, frequency):
        """ return a pandas time series from the past target values with Nan values for the prediction_length future dates """
        target_series = pd.Series(
            np.append(timeseries["target"], np.repeat(np.nan, self.prediction_length)),
            name=timeseries["target_name"],
            index=pd.date_range(
                start=timeseries["start"],
                periods=len(timeseries["target"]) + self.prediction_length,
                freq=frequency,
            ),
        )
        return target_series

    def _generate_history_external_features_dataframe(self, timeseries, frequency):
        """ return a pandas time series from the past and future external features values """
        external_features_df = pd.DataFrame(
            timeseries["feat_dynamic_real"].T[: len(timeseries["target"]) + self.prediction_length],
            columns=timeseries["feat_dynamic_real_columns_names"],
            index=pd.date_range(
                start=timeseries["start"],
                periods=len(timeseries["target"]) + self.prediction_length,
                freq=frequency,
            ),
        )
        return external_features_df

    def _retrieve_history_timeseries(self, frequency):
        """
        compute the history timeseries from the gluon_dataset object and fill the dates to predict with Nan values
        return a dictionary of list of timeseries by identifiers (None if no identifiers)
        """
        history_timeseries = {}
        for i, timeseries in enumerate(self.gluon_dataset.list_data):
            if "identifiers" in timeseries:
                timeseries_identifier_key = tuple(sorted(timeseries["identifiers"].items()))
            else:
                timeseries_identifier_key = None

            target_series = self._generate_history_target_series(timeseries, frequency)

            if "feat_dynamic_real_columns_names" in timeseries:
                assert timeseries["feat_dynamic_real"].shape[1] >= len(timeseries["target"]) + self.prediction_length
                if timeseries_identifier_key not in history_timeseries:
                    external_features_df = self._generate_history_external_features_dataframe(timeseries, frequency)
                    history_timeseries[timeseries_identifier_key] = [external_features_df]

            if timeseries_identifier_key in history_timeseries:
                history_timeseries[timeseries_identifier_key] += [target_series]
            else:
                history_timeseries[timeseries_identifier_key] = [target_series]
        return history_timeseries

    def _compute_forecasts_timeseries(self, forecasts_list):
        """
        compute all forecasts timeseries for each quantile
        return a dictionary of list of forecasts timeseries by identifiers (None if no identifiers)
        """
        forecasts_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            if "identifiers" in self.gluon_dataset.list_data[i]:
                timeseries_identifier_key = tuple(sorted(self.gluon_dataset.list_data[i]["identifiers"].items()))
            else:
                timeseries_identifier_key = None

            for quantile in self.quantiles:
                forecasts_series = (
                    sample_forecasts.quantile_ts(quantile)
                    .rename(
                        "{}_forecasts_percentile_{}".format(
                            self.gluon_dataset.list_data[i]["target_name"],
                            int(quantile * 100),
                        )
                    )
                    .iloc[: self.prediction_length]
                )
                if timeseries_identifier_key in forecasts_timeseries:
                    forecasts_timeseries[timeseries_identifier_key] += [forecasts_series]
                else:
                    forecasts_timeseries[timeseries_identifier_key] = [forecasts_series]
        return forecasts_timeseries

    def _concat_timeseries_per_identifiers(self, all_timeseries):
        """
        concat on columns all forecasts timeseries with same identifiers
        return a list of timeseries with multiple forecasts for each identifiers
        """
        multiple_df = []
        for timeseries_identifier_key, series_list in all_timeseries.items():
            unique_identifiers_df = pd.concat(series_list, axis=1).reset_index(drop=False)
            if timeseries_identifier_key:
                for identifier_key, identifier_value in timeseries_identifier_key:
                    unique_identifiers_df[identifier_key] = identifier_value
            multiple_df += [unique_identifiers_df]
        return multiple_df

    def _concat_all_timeseries(self, multiple_df):
        """ concat on rows all forecasts (one identifiers timeseries after the other) and rename time column """
        return pd.concat(multiple_df, axis=0).reset_index(drop=True)

    def _reorder_forecasts_df(self, time_column_name, identifiers_columns):
        """ reorder columns with timeseries_identifiers just after time column """
        forecasts_columns = [column for column in self.forecasts_df if column not in [time_column_name] + identifiers_columns]
        self.forecasts_df = self.forecasts_df[[time_column_name] + identifiers_columns + forecasts_columns]

    def get_forecasts_df(self, session=None, model_type=None):
        """ add the session timestamp and model_type to forecasts dataframe """
        if session:
            self.forecasts_df["session"] = session
        if model_type:
            self.forecasts_df["model_type"] = model_type
        return self.forecasts_df

    def create_forecasts_column_description(self):
        """ explain the meaning of the forecasts columns """
        column_descriptions = {}
        for column in self.forecasts_df.columns:
            if "_forecasts_percentile_50" in column:
                column_descriptions[column] = "Median of all sample predictions."
            elif "_forecasts_percentile_" in column:
                column_descriptions[column] = "{}% of sample predictions are below these values.".format(column.split("_")[-1])
        return column_descriptions

    def _check(self):
        if self.predictor.prediction_length < self.prediction_length:
            raise ValueError(
                "The selected prediction length ({}) cannot be higher than the one ({}) used in training !".format(
                    self.prediction_length, self.predictor.prediction_length
                )
            )
