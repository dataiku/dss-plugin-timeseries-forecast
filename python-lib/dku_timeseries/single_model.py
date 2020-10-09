import dill as pickle #todo: check whether or not dill still necessary
from gluonts.trainer import Trainer
from plugin_io_utils import get_estimator, write_to_folder
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import pandas as pd
try:
    from BytesIO import BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3
from json import dumps


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
            trainer=Trainer(epochs=self.epoch)  # 10
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
        self.predictor = self.estimator.train(train_ds)

    def evaluate(self, train_ds, test_ds, make_forecasts=False):
        predictor = self.estimator.train(train_ds)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
        ts_list = list(ts_it)
        forecast_list = list(forecast_it)

        agg_metrics, item_metrics = evaluator(iter(ts_list), iter(forecast_list), num_series=len(test_ds))

        item_metrics['item_id'] = self.model_name
        agg_metrics['item_id'] = self.model_name

        target_cols = [time_series['target'].name for time_series in train_ds.list_data]
        item_metrics.insert(1, 'target_col', target_cols)

        # TODO only if multiple target columns
        agg_metrics['target_col'] = 'AGGREGATION'
        item_metrics = item_metrics.append(agg_metrics, ignore_index=True)

        if make_forecasts:
            series = []
            for i, sample_forecasts in enumerate(forecast_list):
                series.append(sample_forecasts.mean_ts.rename("{}_{}".format(target_cols[i], self.model_name)))
            forecasts_df = pd.concat(series, axis=1).reset_index()
            return agg_metrics, item_metrics, forecasts_df

        return agg_metrics, item_metrics

    def save(self, model_folder, version_name):  # todo: review how folder/paths are handled
        model_path = "{}/{}/model.pk".format(version_name, self.model_name)
        write_to_folder(self.predictor, model_folder, model_path, 'pickle')

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
