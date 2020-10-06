from dku_timeseries.single_model import SingleModel
try:
    from BytesIO import BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3


class GlobalModels():
    def __init__(self, global_params, models_parameters):
        self.global_params = global_params
        self.models_parameters = models_parameters
        self.model_names = []
        self.models = None
        self.glutonts_dataset = None

    def init_all_models(self):
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                SingleModel(model_name, model_parameters, self.global_params)
            )  # model_name, model_params, global_models_params

    def fit_all(self, gluonts_dataset):
        for model in self.models:
            model.fit(gluonts_dataset)

    def evaluate_all(self, eval_params, glutonts_dataset):
        models_error = []
        self.glutonts_dataset = glutonts_dataset
        for model in self.models:
            agg_metrics = model.evaluate(eval_params, glutonts_dataset.get_list_dataset())
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
