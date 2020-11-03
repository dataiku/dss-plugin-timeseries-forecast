import pandas as pd
from gluonts.dataset.common import ListDataset
from plugin_config_loading import PluginParamValidationError


class Prediction():
    def __init__(self, predictor, gluon_train_dataset, prediction_length, quantiles, include_history):
        self.predictor = predictor
        self.gluon_train_dataset = gluon_train_dataset
        # self.targets_train_df = targets_train_df
        # self.time_column = targets_train_df.columns[0]
        # self.target_columns = targets_train_df.columns[1:]
        # self.external_features_df = external_features_df
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.include_history = include_history
        # self._check()

    def predict(self):
        # prediction_dataset = self._create_gluonts_dataset()
        # long_format = bool('identifiers' in self.gluon_train_dataset.list_data[0])
        # if long_format and self.include_history:
        #     raise ValueError("Cannot include history in long format")

        # TODO include external feature dataset in self.gluon_train_dataset

        forecasts = self.predictor.predict(self.gluon_train_dataset)
        forecasts_list = list(forecasts)

        all_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            if 'identifiers' in train_ds.list_data[i]:
                timeseries_identifier_key = tuple(sorted(train_ds.list_data[i]['identifiers'].items()))
            else:
                timeseries_identifier_key = None

            for quantile in self.quantiles:
                series = sample_forecasts.quantile_ts(quantile).rename("{}_forecasts_percentile_{}".format(self.gluon_train_dataset.list_data[i]['target_name'], int(quantile*100))))
                if timeseries_identifier_key in all_timeseries:
                    all_timeseries[timeseries_identifier_key] += [series]
                else:
                    all_timeseries[timeseries_identifier_key] = [series]

        multiple_df = []
        for timeseries_identifier_key, series_list in all_timeseries.items():
            unique_identifiers_df = pd.concat(series_list, axis=1).reset_index(drop=False)
            if timeseries_identifier_key:
                for identifier_key, identifier_value in timeseries_identifier_key:
                    unique_identifiers_df[identifier_key] = identifier_value
            multiple_df += [unique_identifiers_df]

        self.forecasts_df = pd.concat(multiple_df, axis=0).reset_index(drop=True).rename(columns={'index': 'time_column'})

        # TODO include history

        # else:
        #     series = []
        #     for i, sample_forecasts in enumerate(forecasts_list):
        #         is_median = False
        #         for quantile in self.quantiles:
        #             # replace percentile_50 with median and always output the median
        #             if quantile == 0.5 or (quantile > 0.5 and not is_median):
        #                 series.append(sample_forecasts.quantile_ts(0.5).rename("{}_forecasts_median".format(self.gluon_train_dataset.list_data[i]['target_name'])))
        #                 is_median = True
        #             if quantile != 0.5:
        #                 series.append(sample_forecasts.quantile_ts(quantile).rename("{}_forecasts_percentile_{}".format(self.gluon_train_dataset.list_data[i]['target_name'], int(quantile*100))))

        #     self.forecasts_df = pd.concat(series, axis=1).reset_index().rename(columns={'index': 'time_column'})

            # # include history
            # if self.include_history:
            #     self.forecasts_df = self.targets_train_df.append(self.forecasts_df).reset_index(drop=True)

            # if self.external_features_df is not None:
            #     self.forecasts_df = self.forecasts_df.merge(self.external_features_df, on=self.time_column, how='left', suffixes=('', '_external_feat'))

            # # only keep the first prediction_length predictions
            # if self.predictor.prediction_length > self.prediction_length:
            #     diff = self.predictor.prediction_length - self.prediction_length
            #     self.forecasts_df = self.forecasts_df.iloc[:-diff]

    def get_forecasts_df(self):
        # TODO add to forecasts dataframe the selected model and session ?
        return self.forecasts_df

    # def _create_gluonts_dataset(self):
    #     freq = self.predictor.freq
    #     start = self.targets_train_df[self.time_column].iloc[0]
    #     if self.external_features_df is not None:
    #         feat_dynamic_real = self.external_features_df.drop(self.time_column, axis=1).values.T
    #         # self.targets_train_df.drop(self.target_columns).append(self.external_features_future_df).drop(self.time_column)
    #         input_ds = ListDataset(
    #             [{
    #                 'target': self.targets_train_df[target_column],
    #                 'feat_dynamic_real': feat_dynamic_real,
    #                 'start': start
    #             } for target_column in self.target_columns],
    #             freq=freq)
    #     else:
    #         input_ds = ListDataset(
    #             [{
    #                 'target': self.targets_train_df[target_column],
    #                 'start': start
    #             } for target_column in self.target_columns],
    #             freq=freq)

    #     return input_ds

    def create_forecasts_column_description(self):
        column_descriptions = {}
        for column in self.forecasts_df.columns:
            if '_forecasts_median' in column:
                column_descriptions[column] = "Median of all sample predictions."
            elif '_forecasts_percentile_' in column:
                column_descriptions[column] = "{}% of sample predictions are below these values.".format(column.split('_')[-1])
        return column_descriptions

    def _check(self):
        if self.predictor.prediction_length < self.prediction_length:
            raise PluginParamValidationError("Forecasting horizon cannot be higher than the prediction length ({}) used in training !".format(
                self.predictor.prediction_length))

        if self.external_features_df is not None:
            if len(self.external_features_df.index) != self.predictor.prediction_length + len(self.targets_train_df.index):
                raise ValueError("External feature dataset must have exactly {} rows (same as prediction_length used in training)".format(
                    self.predictor.prediction_length))
