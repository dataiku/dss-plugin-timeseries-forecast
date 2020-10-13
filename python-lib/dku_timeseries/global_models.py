import pandas as pd
from dku_timeseries.single_model import SingleModel
from gluonts.dataset.common import ListDataset
from plugin_io_utils import write_to_folder


class GlobalModels():
    def __init__(self, target_columns_names, time_column_name, frequency, model_folder, epoch, models_parameters, prediction_length,
                 training_df, make_forecasts, external_features_column_name=None):
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
        self.make_forecasts = make_forecasts
        self.external_features_column_name = external_features_column_name

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
        self.training_df[self.time_col] = pd.to_datetime(self.training_df[self.time_col]).dt.tz_localize(tz=None)
        if self.make_forecasts:
            self.forecasts_df = pd.DataFrame()

    def fit_all(self):
        # create list dataset for fit
        train_ds = self.create_gluonts_dataset(length=len(self.training_df.index))
        for model in self.models:
            model.fit(train_ds)

    def create_gluonts_dataset(self, length):
        initial_date = self.training_df[self.time_col].iloc[0]
        start = pd.Timestamp(initial_date, freq=self.frequency)
        if not self.external_features_column_name:
            return ListDataset(
                [{
                    "start": start,
                    "target": self.training_df[target_column_name].iloc[:length]  # start from 0 to length
                } for target_column_name in self.target_columns_names],
                freq=self.frequency
            )
        else:
            external_features_all_df = self.training_df[self.external_features_column_name].iloc[:length]
            return ListDataset(
                [{
                    'start': start,
                    'target': self.training_df[target_column_name].iloc[:length],  # start from 0 to length
                    'feat_dynamic_real': external_features_all_df.values.T
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

        self.metrics_df = pd.DataFrame()
        for model in self.models:
            if self.make_forecasts:
                agg_metrics, item_metrics, forecasts_df = model.evaluate(train_ds, test_ds, make_forecasts=True)
                forecasts_df = forecasts_df.rename(columns={'index': self.time_col})
                if self.forecasts_df.empty:
                    self.forecasts_df = forecasts_df
                else:
                    self.forecasts_df = self.forecasts_df.merge(forecasts_df, on=self.time_col)
            else:
                agg_metrics, item_metrics = model.evaluate(train_ds, test_ds)
            self.metrics_df = self.metrics_df.append(item_metrics)

    def save_all(self, version_name):
        metrics_path = "{}/metrics.csv".format(version_name)
        write_to_folder(self.metrics_df, self.model_folder, metrics_path, 'csv')

        df_path = "{}/train_dataset.csv.gz".format(version_name)
        write_to_folder(self.training_df, self.model_folder, df_path, 'csv.gz')

        for model in self.models:
            model.save(model_folder=self.model_folder, version_name=version_name)

    def prediction(self, model_name):
        return

    def load(self, path):
        # Todo
        dataset = load(dataset)
        best_model = find_best_model(dataset)
        model = SingleModel()
        model.load(path, best_model)

    def get_history_and_forecasts_df(self):
        return self.training_df.merge(self.forecasts_df, on=self.time_col, how='left')

    def get_metrics_df(self):
        return self.metrics_df
