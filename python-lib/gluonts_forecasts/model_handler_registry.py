from gluonts_forecasts.model_handler import (
    TrivialIdentity,
    SeasonalNaive,
    AutoARIMA,
    SeasonalTrend,
    NPTS,
    FeedForward,
    DeepAR,
    Transformer,
    MQCNN,
)


class ModelHandlerRegistry:
    def __init__(self):
        self.models = {}
        self.register(TrivialIdentity)
        self.register(SeasonalNaive)
        self.register(AutoARIMA)
        self.register(SeasonalTrend)
        self.register(NPTS)
        self.register(FeedForward)
        self.register(DeepAR)
        self.register(Transformer)
        self.register(MQCNN)

    def register(self, model):
        self.models[model.model_name] = model()

    def get_model(self, model_name):
        return self.models.get(model_name)

    def get_model_label(self, model_name):
        model = self.get_model(model_name)
        model_label = model.get_label() if model is not None else None
        return model_label

    def list_available_models(self):
        return [model_name for model_name in self.models]

    def list_available_models_labels(self):
        available_models_labels = []
        for model_name in self.models:
            label = self.get_model_label(model_name)
            if label is not None:
                available_models_labels.append(label)
        return available_models_labels

    def get_model_name_from_label(self, model_label):
        return next((model_name for model_name in self.models if self.get_model_label(model_name) == model_label), None)
