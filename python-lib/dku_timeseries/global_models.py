from dku_timeseries.single_model import SingleModel
from gluonts.dataset.common import ListDataset
try:
    from BytesIO import BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3


class GlobalModels():
    def __init__(self, global_params, models_parameters, training_df):  # df as input ?
        self.global_params = global_params
        self.models_parameters = models_parameters
        self.model_names = []
        self.models = None
        self.glutonts_dataset = None
        self.training_df = training_df

    def init_all_models(self):
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                SingleModel(model_name, model_parameters, self.global_params)
            )  # model_name, model_params, global_models_params

    def fit_all(self):
        # create list dataset for fit
        train_ds = self.create_gluonts_dataset(length=len(self.training_df.index))
        for model in self.models:
            model.fit(train_ds)

    def create_gluonts_dataset(self, length):

        return ListDataset(
            [{
                "start": self.training_df[self.time_col].iloc[0],
                "target": self.training_df[target_col].iloc[:length]  # start from 0 to length
            } for target_col in self.target_cols],
            freq=self.frequency
        )

    def evaluate_all(self, evaluation_strategy):
        total_length = len(self.training_df.index)
        if evaluation_strategy == "split":
            train_ds = self.create_gluonts_dataset(length=total_length-self.prediction_length)  # all - prediction_length time steps
            test_ds = self.create_gluonts_dataset(length=total_length)  # all time steps
        # else:
        #     for window in rolling_windows:
        #         train_ds = create_gluonts_dataset("[0, window * window_size] time steps")
        #         test_ds = create_gluonts_dataset("[0, window * window_size + prediction_length] time steps")


        models_error = []
        # self.glutonts_dataset = glutonts_dataset
        for model in self.models:
            agg_metrics = model.evaluate(train_ds, test_ds)
            models_error.append(agg_metrics)
        return models_error

    def save_all(self, version_name):
        model_folder = self.global_params['model_folder']
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
