from plugin_config_loading import load_predict_config
from plugin_io_utils import set_column_description, check_external_features_future_dataset_schema
from dku_timeseries.model_selection import ModelSelection
from dku_timeseries.prediction import Prediction, add_future_external_features

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

gluon_train_dataset = model_selection.get_gluon_train_dataset()  # => ListDataset()

# TODO handle exception if feat_dynamic_real in training but no external_feat dataset provided (or the opposite)
if params['external_features_future_dataset']:
    check_external_features_future_dataset_schema(gluon_train_dataset, params['external_features_future_dataset'])
    external_features_future_df = params['external_features_future_dataset'].get_dataframe()

gluon_dataset = add_future_external_features(gluon_train_dataset, external_features_future_df)

prediction = Prediction(
    predictor=predictor,
    gluon_dataset=gluon_dataset,
    prediction_length=params['prediction_length'],
    quantiles=params['quantiles'],
    include_history=params['include_history']
)

prediction.predict()

forecasts_df = prediction.get_forecasts_df(session=model_selection.session, model_type=model_selection.model_type)
params['output_dataset'].write_with_schema(forecasts_df)

column_descriptions = prediction.create_forecasts_column_description()
set_column_description(params['output_dataset'], column_descriptions)
