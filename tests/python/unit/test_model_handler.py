from gluonts_forecasts.model_handler import ModelHandler


def test_get_model_descriptor():
    model = ModelHandler("no model")
    assert model._get_model_descriptor() is not None
