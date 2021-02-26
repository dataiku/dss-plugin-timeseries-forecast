from dku_io_utils.recipe_config_loading import load_predict_config
from dku_io_utils.utils import set_column_description
from dku_io_utils.checks_utils import external_features_check
from dku_io_utils.model_selection import ModelSelection
from gluonts_forecasts.utils import add_future_external_features
from gluonts_forecasts.trained_model import TrainedModel
from safe_logger import SafeLogger
from time import perf_counter

logger = SafeLogger("Forecast plugin")
params = load_predict_config()

start = perf_counter()
logger.info("Forecasting future values...")

model_selection = ModelSelection(
    folder=params["model_folder"],
    partition_root=params["partition_root"],
)

if params["manual_selection"]:
    model_selection.set_manual_selection_parameters(session_name=params["selected_session"], model_label=params["selected_model_label"])
else:
    model_selection.set_auto_selection_parameters(performance_metric=params["performance_metric"])

predictor = model_selection.get_model_predictor()

gluon_train_dataset = model_selection.get_gluon_train_dataset()

external_features = external_features_check(gluon_train_dataset, params["external_features_future_dataset"])

if external_features:
    external_features_future_df = params["external_features_future_dataset"].get_dataframe()
    gluon_train_dataset = add_future_external_features(
        gluon_train_dataset,
        external_features_future_df,
        predictor.prediction_length,
        predictor.freq
    )

trained_model = TrainedModel(
    model_name=model_selection.get_model_name(),
    predictor=predictor,
    gluon_dataset=gluon_train_dataset,
    prediction_length=params["prediction_length"],
    quantiles=params["quantiles"],
    include_history=params["include_history"],
    history_length_limit=params["history_length_limit"],
)

trained_model.predict()

logger.info("Forecasting future values: Done in {:.2f} seconds".format(perf_counter() - start))

forecasts_df = trained_model.get_forecasts_df(session=model_selection.get_session_name())
params["output_dataset"].write_with_schema(forecasts_df)

column_descriptions = trained_model.create_forecasts_column_description()
set_column_description(params["output_dataset"], column_descriptions)
