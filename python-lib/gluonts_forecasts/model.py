from constants import EVALUATION_METRICS, METRICS_DATASET
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import pandas as pd
import logging
from gluonts_forecasts.model_descriptor import ModelDescriptor


class Model:
    def __init__(
        self,
        model_name,
        model_parameters,
        frequency,
        prediction_length,
        epoch,
        use_external_features=False,
        batch_size=None,
        gpu=None,
        context_length=None,
    ):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model_descriptor = ModelDescriptor(model_name)
        self.frequency = frequency
        self.prediction_length = prediction_length
        self.epoch = epoch
        self.use_external_features = use_external_features
        estimator_kwargs = {
            "freq": self.frequency,
            "prediction_length": self.prediction_length,
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
        self.estimator = self.model_descriptor.get_estimator(self.model_parameters, **estimator_kwargs)
        self.predictor = None
        if self.estimator is None:
            print("{} model is not implemented yet".format(model_name))

    def get_name(self):
        return self.model_name

    def train(self, train_list_dataset):
        if self.estimator is None:
            print("No training: {} model not implemented yet".format(self.model_name))
            return
        logging.info("Timeseries forecast - Training model {} on all data".format(self.model_name))
        self.predictor = self.estimator.train(train_list_dataset)

    def evaluate(self, train_list_dataset, test_list_dataset, make_forecasts=False):
        """
        train model on train_list_dataset and evaluate it on test_list_dataset
        return evaluation metrics for each target and aggregated
        return a dataframe of predictions if make_forecasts is True
        """
        logging.info("Timeseries forecast - Training model {} for evaluation".format(self.model_name))
        evaluation_predictor = self._get_predictor(train_list_dataset)

        agg_metrics, item_metrics, forecasts = self._make_evaluation_predictions(evaluation_predictor, test_list_dataset)

        metrics, identifiers_columns = self._format_metrics(agg_metrics, item_metrics, train_list_dataset)

        if make_forecasts:
            forecasts_df = self._get_forecasts_df(forecasts, train_list_dataset)
            return metrics, identifiers_columns, forecasts_df

        return metrics, identifiers_columns

    def _make_evaluation_predictions(self, predictor, test_list_dataset):
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_list_dataset, predictor=predictor, num_samples=100)
        timeseries = list(ts_it)
        forecasts = list(forecast_it)
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts), num_series=len(test_list_dataset))
        return agg_metrics, item_metrics, forecasts

    def _format_metrics(self, agg_metrics, item_metrics, train_list_dataset):
        """
        return a metrics dataframe with both item_metrics and agg_metrics concatenated and the identifiers columns
        and add new columns: model_name, target_column, identifiers_columns
        """
        item_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_name
        agg_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_name

        identifiers_columns = list(train_list_dataset.list_data[0]["identifiers"].keys()) if "identifiers" in train_list_dataset.list_data[0] else []
        identifiers_values = {identifiers_column: [] for identifiers_column in identifiers_columns}
        target_columns = []
        for univariate_timeseries in train_list_dataset.list_data:
            target_columns += [univariate_timeseries["target_name"]]
            for identifiers_column in identifiers_columns:
                identifiers_values[identifiers_column] += [univariate_timeseries["identifiers"][identifiers_column]]

        item_metrics[METRICS_DATASET.TARGET_COLUMN] = target_columns
        agg_metrics[METRICS_DATASET.TARGET_COLUMN] = METRICS_DATASET.AGGREGATED_ROW

        for identifiers_column in identifiers_columns:
            item_metrics[identifiers_column] = identifiers_values[identifiers_column]
            agg_metrics[identifiers_column] = METRICS_DATASET.AGGREGATED_ROW  # or keep empty but will cast integer to float

        metrics = item_metrics.append(agg_metrics, ignore_index=True)

        metrics = metrics[[METRICS_DATASET.TARGET_COLUMN] + identifiers_columns + [METRICS_DATASET.MODEL_COLUMN] + EVALUATION_METRICS]
        metrics["model_params"] = self._get_model_parameters_json()

        return metrics, identifiers_columns

    def _get_predictor(self, train_list_dataset):
        if self.estimator is None:
            predictor = self.model_descriptor.get_predictor(freq=self.frequency, prediction_length=self.prediction_length)
        else:
            predictor = self.estimator.train(train_list_dataset)
        return predictor

    def _get_model_parameters_json(self):
        return str(
            {
                "model_name": self.model_name,
                "model_parameters": self.model_parameters,
                "frequency": self.frequency,
                "prediction_length": self.prediction_length,
                "epoch": self.epoch,
                "use_external_features": self.use_external_features,
                "batch_size": self.batch_size,
            }
        )

    def _get_forecasts_df(self, forecasts_list, train_list_dataset):
        all_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            series = sample_forecasts.mean_ts.rename("{}_{}".format(train_list_dataset.list_data[i]["target_name"], self.model_name))
            if "identifiers" in train_list_dataset.list_data[i]:
                timeseries_identifier_key = tuple(sorted(train_list_dataset.list_data[i]["identifiers"].items()))
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