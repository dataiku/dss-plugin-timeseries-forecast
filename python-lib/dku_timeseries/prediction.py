import pandas as pd
from gluonts.dataset.common import ListDataset
from plugin_config_loading import PluginParamValidationError


class Prediction():
    def __init__(self, predictor, training_df, external_features_future_df, time_col, target_cols, forecasting_horizon, quantiles):
        self.predictor = predictor
        self.training_df = training_df
        self.external_features_future_df = external_features_future_df
        self.time_col = time_col
        self.target_cols = target_cols
        self.forecasting_horizon = forecasting_horizon
        self.quantiles = quantiles
        self._check()

    def predict(self):
        prediction_dataset = self._create_gluonts_dataset()

        forecasts = self.predictor.predict(prediction_dataset)
        forecasts_list = list(forecasts)

        series = []
        for i, sample_forecasts in enumerate(forecasts_list):
            # series.append(sample_forecasts.mean_ts.rename("{}_mean_forecasts".format(self.target_cols[i])))
            for quantile in self.quantiles:
                series.append(sample_forecasts.quantile_ts(quantile).rename("{}_forecasts_percentile_{}".format(self.target_cols[i], int(quantile*100))))

        predictions_df = pd.concat(series, axis=1).reset_index().rename(columns={'index': self.time_col})

        if self.external_features_future_df:
            predictions_df = predictions_df.merge(self.external_features_future_df, on=self.time_col)

        # only keep the first forecasting_horizon predictions
        predictions_df = predictions_df.iloc[:self.forecasting_horizon]

        # include history
        self.results_df = self.training_df.append(predictions_df)

    def get_results_dataframe(self):
        return self.results_df

    def _create_gluonts_dataset(self):
        freq = self.predictor.freq
        start = self.training_df[self.time_col].iloc[0]
        if not self.external_features_future_df:
            input_ds = ListDataset(
                [{
                    'target': self.training_df[target_col],
                    'start': start
                } for target_col in self.target_cols],
                freq=freq)
        else:
            feat_dynamic_real_df = self.training_df.drop(self.target_cols).append(self.external_features_future_df).drop(self.time_col)
            input_ds = ListDataset(
                [{
                    'target': self.training_df[target_col],
                    'feat_dynamic_real': feat_dynamic_real_df.values.T,
                    'start': start
                } for target_col in self.target_cols],
                freq=freq)

        return input_ds

    def _check(self):
        if self.predictor.prediction_length < self.forecasting_horizon:
            raise PluginParamValidationError("Forecasting horizon cannot be higher than the prediction length ({}) used in training !".format(
                self.predictor.prediction_length))

        try:
            self.training_df[self.time_col] = pd.to_datetime(self.training_df[self.time_col]).dt.tz_localize(tz=None)
        except Exception as e:
            raise ValueError("Could not convert to datetime: {}".format(e))

        if self.external_features_future_df:
            if len(self.external_features_future_df.index) != self.predictor.prediction_length:
                raise ValueError("External feature dataset must have exactly {} rows (same as prediction_length used in training)".format(
                    self.predictor.prediction_length))

            if set(self.training_df.columns.drop(self.target_cols)) == set(self.external_features_future_df.columns):
                raise ValueError("External feature dataset must only contain the following columns: {}".format(
                    self.training_df.columns.drop(self.target_cols)))

            # maybe also check for identical dtypes
            # maybe check that external_features_future_df just follow training_df (based on time_col)
