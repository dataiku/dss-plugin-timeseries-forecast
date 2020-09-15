from single_model import SingleModel


class GlobalModels():
    def __init__(self, global_params):
        self.global_params = global_params

        self.model_names = []

    def init_all_models(self):
        self.models = []
        for model_name in self.global_params.models_names:
            self.models.append(SingleModel(model_name, self.global_params.get_model_params(model_name), self.global_params.get_global_model_params()))

    def fit_all(self, gluonts_dataset):
        for model in self.models:
            model.fit(gluonts_dataset)

    def evaluate_all(self, eval_params, glutonts_dataset):
        for model in self.models:
            model.evaluate(eval_params, glutonts_dataset)

    def save_all(self, path):
        for model in self.models:
            model.save(path="{}/{}".format(path, model_name))
        save(dataset)  # csv with results

    def prediction(self, model_name):
        return

    def load(self, path):
        dataset = load(dataset)
        best_model = find_best_model(dataset)
        model = SingleModel()
        model.load(path, best_model)
