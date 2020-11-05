import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
import copy
from functools import reduce


def apply_filter_conditions(df, conditions):
    """
    return a function to apply filtering conditions on df
    """
    if len(conditions) == 0:
        return df
    elif len(conditions) == 1:
        return df[conditions[0]]
    else:
        return df[reduce(lambda c1, c2: c1 & c2, conditions[1:], conditions[0])]


def add_future_external_features(gluon_train_dataset, external_features_future_df, prediction_length):
    """ append the future external features to the gluonTS ListDataset used for training """
    gluon_dataset = copy.deepcopy(gluon_train_dataset)
    for i, timeseries in enumerate(gluon_train_dataset):
        if 'identifiers' in timeseries:
            timeseries_identifiers = timeseries['identifiers']
            conditions = [external_features_future_df[k]==v for k, v in timeseries_identifiers.items()]
            timeseries_external_features_future_df = apply_filter_conditions(external_features_future_df, conditions)
        else:    
            timeseries_external_features_future_df = external_features_future_df

        feat_dynamic_real_train = timeseries['feat_dynamic_real']
        feat_dynamic_real_columns_names = timeseries['feat_dynamic_real_columns_names']

        feat_dynamic_real_future = timeseries_external_features_future_df[feat_dynamic_real_columns_names].values.T

        if feat_dynamic_real_future.shape[1] != prediction_length:
            raise ValueError("Length of future external features timeseries must be equal to the training prediction length ({})".format(prediction_length))

        feat_dynamic_real_appended = np.append(feat_dynamic_real_train, feat_dynamic_real_future, axis=1)

        gluon_dataset.list_data[i]['feat_dynamic_real'] = feat_dynamic_real_appended
        
    return gluon_dataset


class Prediction():
    def __init__(self, predictor, gluon_dataset, prediction_length, quantiles, include_history):
        self.predictor = predictor
        self.gluon_dataset = gluon_dataset
        self.prediction_length = predictor.prediction_length if prediction_length == 0 else prediction_length
        self.quantiles = quantiles
        self.include_history = include_history # TODO ? implement include history
        self.forecasts_df = None
        self._check()

    def predict(self):
        """
        use the gluon dataset of training to predict future values and
        concat all forecasts timeseries of different identifiers and quantiles together
        """
        forecasts = self.predictor.predict(self.gluon_dataset)
        forecasts_list = list(forecasts)

        all_timeseries = self._compute_all_forecasts_timeseries(forecasts_list)

        multiple_df = self._concat_all_forecasts_timeseries_per_identifiers(all_timeseries)

        time_column_name = self.gluon_dataset.list_data[0]['time_column_name']

        self.forecasts_df = self._concat_all_forecasts_timeseries(multiple_df, time_column_name)

        if 'identifiers' in self.gluon_dataset.list_data[0]:
            self._reorder_forecasts_df(time_column_name)

    def _compute_all_forecasts_timeseries(self, forecasts_list):
        """
        compute all forecasts timeseries for each quantile
        return a dictionary of list of forecasts timeseries by identifiers (None if no identifiers)
        """
        all_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            if 'identifiers' in self.gluon_dataset.list_data[i]:
                timeseries_identifier_key = tuple(sorted(self.gluon_dataset.list_data[i]['identifiers'].items()))
            else:
                timeseries_identifier_key = None

            for quantile in self.quantiles:
                forecasts_series = sample_forecasts.quantile_ts(quantile).rename(
                    "{}_forecasts_percentile_{}".format(self.gluon_dataset.list_data[i]['target_name'], int(quantile*100))
                ).iloc[:self.prediction_length]
                if timeseries_identifier_key in all_timeseries:
                    all_timeseries[timeseries_identifier_key] += [forecasts_series]
                else:
                    all_timeseries[timeseries_identifier_key] = [forecasts_series]
        return all_timeseries

    def _concat_all_forecasts_timeseries_per_identifiers(self, all_timeseries):
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

    def _concat_all_forecasts_timeseries(self, multiple_df, time_column_name):
        """ concat on rows all forecasts (one identifiers timeseries after the other) and rename time column """
        return pd.concat(multiple_df, axis=0).reset_index(drop=True).rename(columns={'index': time_column_name})

    def _reorder_forecasts_df(self, time_column_name):
        """ reorder columns with timeseries_identifiers just after time column """
        timeseries_identifiers_columns = list(self.gluon_dataset.list_data[0]['identifiers'].keys())
        forecasts_columns = [column for column in self.forecasts_df if column not in [time_column_name] + timeseries_identifiers_columns]
        self.forecasts_df = self.forecasts_df[[time_column_name] + timeseries_identifiers_columns + forecasts_columns]

    def get_forecasts_df(self, session=None, model_type=None):
        """ add the session timestamp and model_type to forecasts dataframe """
        if session:
            self.forecasts_df['session'] = session
        if model_type:
            self.forecasts_df['model_type'] = model_type
        return self.forecasts_df

    def create_forecasts_column_description(self):
        """ explain the meaning of the forecasts columns """
        column_descriptions = {}
        for column in self.forecasts_df.columns:
            if '_forecasts_median' in column:
                column_descriptions[column] = "Median of all sample predictions."
            elif '_forecasts_percentile_' in column:
                column_descriptions[column] = "{}% of sample predictions are below these values.".format(column.split('_')[-1])
        return column_descriptions

    def _check(self):
        if self.predictor.prediction_length < self.prediction_length:
            raise ValueError("The selected prediction length ({}) cannot be higher than the one ({}) used in training !".format(
                self.prediction_length,
                self.predictor.prediction_length
                )
            )
