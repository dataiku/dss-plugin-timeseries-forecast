import dill as pickle #todo: check whether or not dill still necessary
from gluonts.trainer import Trainer
from plugin_io_utils import get_estimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import backtest_metrics
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

    def evaluate(self, train_ds, test_ds):
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        agg_metrics, item_metrics = backtest_metrics(
            train_dataset=train_ds,
            test_dataset=test_ds,
            forecaster=self.estimator,
            evaluator=evaluator,
            num_samples=100
        )
        item_metrics['item_id'] = self.model_name
        agg_metrics['item_id'] = self.model_name

        return agg_metrics, item_metrics

    def evaluate_and_forecast(self, train_ds, test_ds):
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

        return agg_metrics, item_metrics, forecasts_df

    def save(self, model_folder, version_name): # todo: review how folder/paths are handled
        bytes_io = BytesIO()
        pickle.dump(self.predictor, bytes_io)
        bytes_io.seek(0)
        pickle_file_path = "{}/{}/model.pk".format(version_name, self.model_name)
        parameters_file_path = "{}/{}/params.json".format(version_name, self.model_name)
        model_folder.upload_stream(pickle_file_path, bytes_io)
        json_dump = dumps(self.model_parameters)
        model_folder.upload_stream(parameters_file_path, json_dump)

# file structure:
# Subfolder per timestamp (each time the recipe is run)
# -> CSV with all model results (same as output dataset)
# -> 1 subfolder per model
#   -> model.pk (Predictor object = estimator.train output)
#   -> params.json (local and global params, including external features)
# model_folder/versions/ts/output.csv
# model_folder/versions/ts/model-blaa/model.pk
# model_folder/versions/ts/model-blaa/params.json
