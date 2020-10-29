# import dill as pickle #todo: check whether or not dill still necessary
#  from gluonts.trainer import Trainer
from plugin_io_utils import get_estimator, write_to_folder, EVALUATION_METRICS, get_trainer, METRICS_DATASET
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import pandas as pd
import logging


class SingleModel():
    def __init__(self, model_name, model_parameters, frequency, prediction_length, epoch):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.frequency = frequency
        self.prediction_length = prediction_length
        self.epoch = epoch
        self.estimator = get_estimator(
            self.model_name,
            self.model_parameters,
            freq=self.frequency,
            prediction_length=self.prediction_length,  # 10,
            trainer=get_trainer(model_name, epochs=self.epoch)  # 10
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
        logging.info("Timeseries forecast - Training model {} for evaluation".format(self.model_name))
        predictor = self.estimator.train(train_ds)
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
            item_metrics[identifiers_column] = identifiers_values[identifiers_column]
            # agg_metrics[identifiers_column] = METRICS_DATASET.AGGREGATED_ROW

        item_metrics = item_metrics.append(agg_metrics, ignore_index=True)

        item_metrics = item_metrics[
            [METRICS_DATASET.TARGET_COLUMN] + identifiers_columns + [METRICS_DATASET.MODEL_COLUMN] + EVALUATION_METRICS
        ]
        # TODO add params of model to item_metrics dataframe

        if make_forecasts:
            multiple_df = []
            for i, sample_forecasts in enumerate(forecasts_list):
                sample_forecasts_df = sample_forecasts.mean_ts.to_frame().reset_index().rename(
                    columns={
                        'index': 'time_column',
                        0: f"{train_ds.list_data[i]['target_name']}_{self.model_name}"
                    }
                )
                if len(identifiers_columns) > 0:
                    for identifier, value in train_ds.list_data[i]['identifiers'].items():
                        sample_forecasts_df[identifier] = value
                multiple_df.append(sample_forecasts_df)
            forecasts_df = pd.concat(multiple_df, axis=0).reset_index(drop=True)

            forecasts_df = forecasts_df.groupby(identifiers_columns + ['time_column']).max().reset_index(drop=False)

            # series = []
            # forecasts_columns = ['time_column'] + identifiers_columns + [f"{name}_{self.model_name}" for name in target_columns]
            # forecasts_df = pd.DataFrame(columns=forecasts_columns)
            # for i, sample_forecasts in enumerate(forecasts_list):
            #     train_ds.list_data[i]
            #     series += [sample_forecasts.mean_ts.rename("{}_{}".format(target_columns[i], self.model_name))]
            # forecasts_df = pd.concat(series, axis=1).reset_index()
            return agg_metrics, item_metrics, forecasts_df, identifiers_columns

        return agg_metrics, item_metrics

    def save(self, model_folder, version_name):  # todo: review how folder/paths are handled
        model_path = "{}/{}/model.pk.gz".format(version_name, self.model_name)
        write_to_folder(self.predictor, model_folder, model_path, 'pickle.gz')

        parameters_path = "{}/{}/params.json".format(version_name, self.model_name)
        write_to_folder(self.model_parameters, model_folder, parameters_path, 'json')

# file structure:
# Subfolder per timestamp (each time the recipe is run)
# -> CSV with all model results (same as output dataset)
# -> 1 subfolder per model
#   -> model.pk (Predictor object = estimator.train output)
#   -> params.json (local and global params, including external features)
# model_folder/versions/ts/output.csv
# model_folder/versions/ts/model-blaa/model.pk
# model_folder/versions/ts/model-blaa/params.json
