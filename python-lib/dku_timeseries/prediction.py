import pandas as pd
from gluonts.dataset.common import ListDataset
from plugin_config_loading import PluginParamValidationError


class Prediction():
    def __init__(self, predictor, targets_train_df, external_features_df, forecasting_horizon, quantiles):
        self.predictor = predictor
        self.targets_train_df = targets_train_df
        self.time_column = targets_train_df.columns[0]
        self.target_columns = targets_train_df.columns[1:]
        self.external_features_df = external_features_df
        self.forecasting_horizon = forecasting_horizon
        self.quantiles = quantiles
        self._check()

    def predict(self):
        prediction_dataset = self._create_gluonts_dataset()

        forecasts = self.predictor.predict(prediction_dataset)
        forecasts_list = list(forecasts)

        series = []
        for i, sample_forecasts in enumerate(forecasts_list):
            is_median = False
            for quantile in self.quantiles:
                # replace percentile_50 with median and always output the median
                if quantile == 0.5 or (quantile > 0.5 and not is_median):
                    series.append(sample_forecasts.quantile_ts(0.5).rename("{}_forecasts_median".format(self.target_columns[i])))
                    is_median = True
                if quantile != 0.5:
                    series.append(sample_forecasts.quantile_ts(quantile).rename("{}_forecasts_percentile_{}".format(self.target_columns[i], int(quantile*100))))

        predictions_df = pd.concat(series, axis=1).reset_index().rename(columns={'index': self.time_column})

        # include history
        results_df = self.targets_train_df.append(predictions_df)

        if self.external_features_df:
            results_df = results_df.merge(self.external_features_df, on=self.time_column)

        # only keep the first forecasting_horizon predictions
        if self.predictor.prediction_length > self.forecasting_horizon:
            diff = self.predictor.prediction_length - self.forecasting_horizon
            results_df = results_df.iloc[:-diff]

        return results_df

    def _create_gluonts_dataset(self):
        freq = self.predictor.freq
        start = self.targets_train_df[self.time_column].iloc[0]
        if self.external_features_df:
            # feat_dynamic_real_df = self.targets_train_df.drop(self.target_columns).append(self.external_features_future_df).drop(self.time_column)
            input_ds = ListDataset(
                [{
                    'target': self.targets_train_df[target_column],
                    'feat_dynamic_real': self.external_features_df.values.T,
                    'start': start
                } for target_column in self.target_columns],
                freq=freq)
        else:
            input_ds = ListDataset(
                [{
                    'target': self.targets_train_df[target_column],
                    'start': start
                } for target_column in self.target_columns],
                freq=freq)

        return input_ds

    def _check(self):
        if self.predictor.prediction_length < self.forecasting_horizon:
            raise PluginParamValidationError("Forecasting horizon cannot be higher than the prediction length ({}) used in training !".format(
                self.predictor.prediction_length))

        try:
            self.targets_train_df[self.time_column] = pd.to_datetime(self.targets_train_df[self.time_column]).dt.tz_localize(tz=None)
        except Exception as e:
            raise ValueError("Could not convert to datetime: {}".format(e))

        if self.external_features_df:
            if len(self.external_features_df.index) != self.predictor.prediction_length + len(self.targets_train_df.index):
                raise ValueError("External feature dataset must have exactly {} rows (same as prediction_length used in training)".format(
                    self.predictor.prediction_length))

            # if set(self.targets_train_df.columns.drop(self.target_columns)) == set(self.external_features_future_df.columns):
            #     raise ValueError("External feature dataset must only contain the following columns: {}".format(
            #         self.targets_train_df.columns.drop(self.target_columns)))

            # maybe also check for identical dtypes
            # maybe check that external_features_future_df just follow targets_train_df (based on time_column)
