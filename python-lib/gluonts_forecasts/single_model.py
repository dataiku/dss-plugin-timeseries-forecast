from constants import EVALUATION_METRICS, METRICS_DATASET
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import pandas as pd
import logging
from gluonts_forecasts.model_descriptor import ModelDescriptor
from dku_io_utils.dku_io_utils import write_to_folder


class SingleModel():
    def __init__(self, model_name, model_parameters, frequency, prediction_length, epoch, use_external_features=False,
                 batch_size=None, gpu=None, context_length=None):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model_descriptor = ModelDescriptor(model_name)
        self.frequency = frequency
        self.prediction_length = prediction_length
        self.epoch = epoch
        self.use_external_features = use_external_features
        estimator_kwargs = {
            "freq": self.frequency,
            "prediction_length": self.prediction_length
        }
        trainer_kwargs = {"epochs": self.epoch}
        self.batch_size = batch_size
        if self.batch_size is not None:
            trainer_kwargs.update({"batch_size": self.batch_size})
        trainer = self.model_descriptor.get_trainer(**trainer_kwargs)
        if trainer is not None:
            estimator_kwargs.update({"trainer": trainer})
        if self.model_descriptor.can_use_external_feature() and self.use_external_features:
            estimator_kwargs.update({"use_feat_dynamic_real": True})
        if context_length is not None:
            estimator_kwargs.update({"context_length": context_length})
        self.estimator = self.model_descriptor.get_estimator(
            self.model_parameters,
            **estimator_kwargs
        )
        self.predictor = None
        if self.estimator is None:
            print("{} model is not implemented yet".format(model_name))

    def get_name(self):
        return self.model_name

    def fit(self, train_ds):
        if self.estimator is None:
            print("No training: {} model not implemented yet".format(self.model_name))
            return
        logging.info("Timeseries forecast - Training model {} on all data".format(self.model_name))
        self.predictor = self.estimator.train(train_ds)

    def evaluate(self, train_ds, test_ds, make_forecasts=False):
        # TODO split into multiple functions 
        logging.info("Timeseries forecast - Training model {} for evaluation".format(self.model_name))
        predictor = self._get_predictor(train_ds)
        evaluator = Evaluator()

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
        ts_list = list(ts_it)
        forecasts_list = list(forecast_it)

        agg_metrics, item_metrics = evaluator(iter(ts_list), iter(forecasts_list), num_series=len(test_ds))

        item_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_name
        agg_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_name

        identifiers_columns = list(train_ds.list_data[0]['identifiers'].keys()) if 'identifiers' in train_ds.list_data[0] else []
        identifiers_values = {identifiers_column: [] for identifiers_column in identifiers_columns}
        target_columns = []
        for univariate_timeseries in train_ds.list_data:
            target_columns += [univariate_timeseries['target_name']]
            for identifiers_column in identifiers_columns:
                identifiers_values[identifiers_column] += [univariate_timeseries['identifiers'][identifiers_column]]

        item_metrics[METRICS_DATASET.TARGET_COLUMN] = target_columns
        agg_metrics[METRICS_DATASET.TARGET_COLUMN] = METRICS_DATASET.AGGREGATED_ROW

        for identifiers_column in identifiers_columns:
            # TODO ? integer are casted to float because of missing values in the 'AGGREGATED' rows
            item_metrics[identifiers_column] = identifiers_values[identifiers_column]
            # agg_metrics[identifiers_column] = METRICS_DATASET.AGGREGATED_ROW

        item_metrics = item_metrics.append(agg_metrics, ignore_index=True)

        item_metrics = item_metrics[
            [METRICS_DATASET.TARGET_COLUMN] + identifiers_columns + [METRICS_DATASET.MODEL_COLUMN] + EVALUATION_METRICS
        ]
        item_metrics['model_params'] = self._get_model_parameters_json()

        if make_forecasts:
            forecasts_df = self._get_forecasts_df(forecasts_list, train_ds)
            return agg_metrics, item_metrics, forecasts_df, identifiers_columns

        return agg_metrics, item_metrics

    def _get_predictor(self, train_ds):
        if self.estimator is None:
            predictor = self.model_descriptor.get_predictor(freq=self.frequency, prediction_length=self.prediction_length)
        else:
            predictor = self.estimator.train(train_ds)
        return predictor

    def _get_model_parameters_json(self):
        return str({
            'model_name': self.model_name,
            'model_parameters': self.model_parameters,
            'frequency': self.frequency,
            'prediction_length': self.prediction_length,
            'epoch': self.epoch,
            'use_external_features': self.use_external_features,
            'batch_size': self.batch_size
        })

    def _get_forecasts_df(self, forecasts_list, train_ds):
        all_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            series = sample_forecasts.mean_ts.rename("{}_{}".format(train_ds.list_data[i]['target_name'], self.model_name))
            if 'identifiers' in train_ds.list_data[i]:
                timeseries_identifier_key = tuple(sorted(train_ds.list_data[i]['identifiers'].items()))
            else:
                timeseries_identifier_key = None

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
        forecasts_df = pd.concat(multiple_df, axis=0).reset_index(drop=True)
        return forecasts_df

    def save(self, model_folder, version_name):
        # TODO ? move outside of the class as it interacts with dataiku.Folder objects
        model_path = "{}/{}/model.pk.gz".format(version_name, self.model_name)
        write_to_folder(self.predictor, model_folder, model_path, 'pickle.gz')

        parameters_path = "{}/{}/params.json".format(version_name, self.model_name)
        write_to_folder(self.model_parameters, model_folder, parameters_path, 'json')
