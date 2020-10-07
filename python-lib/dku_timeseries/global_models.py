import pandas as pd
from dku_timeseries.single_model import SingleModel
from gluonts.dataset.common import ListDataset
try:
    from BytesIO import BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3


class GlobalModels():
    def __init__(self, target_columns_names, time_column_name, frequency, model_folder, epoch, models_parameters, prediction_length, training_df, forecast):
        self.models_parameters = models_parameters
        self.model_names = []
        self.models = None
        self.glutonts_dataset = None
        self.training_df = training_df
        self.prediction_length = prediction_length
        self.target_columns_names = target_columns_names
        self.time_col = time_column_name
        self.frequency = frequency
        self.model_folder = model_folder
        self.epoch = epoch
        self.forecast = True if forecast else False

    def init_all_models(self):
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                SingleModel(
                    model_name,
                    model_parameters=model_parameters,
                    frequency=self.frequency,
                    prediction_length=self.prediction_length,
                    epoch=self.epoch
                )
            )

    def fit_all(self):
        # create list dataset for fit
        train_ds = self.create_gluonts_dataset(length=len(self.training_df.index))
        for model in self.models:
            model.fit(train_ds)

    def create_gluonts_dataset(self, length):
        initial_date = self.training_df[self.time_col].iloc[0]
        start = pd.Timestamp(initial_date, freq=self.frequency).tz_localize(None)
        return ListDataset(
            [{
                "start": start,
                "target": self.training_df[target_column_name].iloc[:length]  # start from 0 to length
            } for target_column_name in self.target_columns_names],
            freq=self.frequency
        )

    def evaluate_all(self, evaluation_strategy):
        total_length = len(self.training_df.index)
        if evaluation_strategy == "split":
            train_ds = self.create_gluonts_dataset(length=total_length-self.prediction_length)  # all - prediction_length time steps
            test_ds = self.create_gluonts_dataset(length=total_length)  # all time steps
        else:
            raise Exception("{} evaluation strategy not implemented".format(evaluation_strategy))

        metrics_df = pd.DataFrame()
        for model in self.models:
            if self.forecast:
                agg_metrics, item_metrics, forecasts_df = model.evaluate_and_forecast(train_ds, test_ds)
                # TODO concat forecasts_df to others forecast_df and make it a class field
            else:
                agg_metrics, item_metrics = model.evaluate(train_ds, test_ds)

            item_metrics.insert(1, 'target_col', self.target_columns_names)
            agg_metrics['target_col'] = 'All'
            item_metrics = item_metrics.append(agg_metrics, ignore_index=True)

            metrics_df = metrics_df.append(item_metrics)
        return metrics_df

    def save_all(self, version_name):
        model_folder = self.model_folder
        for model in self.models:
            model.save(model_folder=model_folder, version_name=version_name)

    def prediction(self, model_name):
        return

    def load(self, path):
        # Todo
        dataset = load(dataset)
        best_model = find_best_model(dataset)
        model = SingleModel()
        model.load(path, best_model)

    def get_forecast_df(self):
    # TODO method to output all forecast and true values if self.forecast is True
