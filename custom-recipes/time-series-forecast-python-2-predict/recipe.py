from plugin_config_loading import load_predict_config
from plugin_io_utils import set_column_description
from dku_timeseries.model_selection import ModelSelection
from dku_timeseries.prediction import Prediction

params = load_predict_config()

model_selection = ModelSelection(
    folder=params['model_folder'],
    external_features_future_dataset=params['external_features_future_dataset']
)

if params['manual_selection']:
    model_selection.manual_params(session=params['selected_session'], model_type=params['selected_model_type'])
else:
    model_selection.auto_params(performance_metric=params['performance_metric'])

predictor = model_selection.get_model()  # => Predictor()

targets_train_df = model_selection.get_targets_train_dataframe()  # => DataFrame()
external_features_df = model_selection.get_external_features_dataframe()  # => DataFrame()

prediction = Prediction(
    predictor=predictor,
    targets_train_df=targets_train_df,
    external_features_df=external_features_df,
    forecasting_horizon=params['forecasting_horizon'],
    quantiles=params['quantiles']
)

output_df = prediction.predict()

output_dataset = params['output_dataset']
output_dataset.write_with_schema(output_df)

set_column_description(output_dataset)
