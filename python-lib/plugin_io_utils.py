
import re
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.npts import NPTSEstimator
from gluonts.model.transformer import TransformerEstimator
#from gluonts.model.seasonal_naive import SeasonalNaivePredictor
#from gluonts.model.naive_2 import Naive2Predictor
import os
import dill as pickle
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import backtest_metrics
import pandas as pd

import dataiku
try:
    from io import StringIO as BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3


AVAILABLE_MODELS = [
    "naive", "simplefeedforward", "deepfactor", "deepar", "lstnet", "nbeats",
    "npts", "transformer"
]


def read_pickle_from_folder(path, folder):
    with folder.get_download_stream(path) as stream:
        data = stream.read()
        return pickle.loads(data)


def write_object_as_pickle_to_folder(obj, path, folder):
    with folder.get_writer(path) as writer:
        pickled = pickle.dumps(obj)
        writer.write(pickled)


def write_dataframe_as_csv_to_folder(df, path, folder):
    with folder.get_writer(path=path) as writer:
        string_buf = df.to_csv(sep=',', na_rep='', header=True, index=False)
        writer.write(string_buf.encode())


def read_csv_from_folder(path, folder):
    """ return dataframe from local path in folder """
    with folder.get_download_stream(path) as stream:
        data = stream.read()
        data = io.StringIO(data.decode())
    return pd.read_csv(data, sep=",", compression='infer')


def get_models_parameters(config):
    models_parameters = {}
    for model in AVAILABLE_MODELS:
        if is_activated(config, model):
            models_parameters.update({
                model: get_model_presets(config, model)
            })
    return models_parameters


def is_activated(config, model):
    return config.get("{}_model_activated".format(model), False)


def get_model_kwargs(config, model):
    return config.get("{}_model_kwargs")


def get_model_presets(config, model):
    model_presets = {}
    matching_key = "{}_model_(.*)".format(model)
    for key in config:
        key_search = re.match(matching_key, key, re.IGNORECASE)
        if key_search:
            key_type = key_search.group(1)
            model_presets.update({
                key_type: config[key]
            })
    return model_presets


def get_estimator(model, model_parameters, **kwargs):
    # "naive", "simplefeedforward", "deepfactor", "deepar", "lstnet", "nbeats",
    # "npts", "transformer"
    # SeasonalNaivePredictor
    # Naive2Predictor
    kwargs.update(model_parameters.get("kwargs", {}))
    if model == "simplefeedforward":
        return SimpleFeedForwardEstimator(**kwargs)
    if model == "deepfactor":
        return DeepFactorEstimator(**kwargs)
    if model == "deepar":
        return DeepAREstimator(**kwargs)
    if model == "lstnet":
        return LSTNetEstimator(**kwargs)
    if model == "nbeats":
        return NBEATSEstimator(**kwargs)
    if model == "npts":
        return NPTSEstimator(**kwargs)
    if model == "transformer":
        return TransformerEstimator(**kwargs)
    return None


def save_forecasting_objects(folder_name, version_name, forecasting_object):
    #filename = "predictor_{}.pk".format(version_name)
    path = os.path.join(folder_name, "versions", version_name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "models.pk"), 'wb') as predictor_file:
        pickle.dump(forecasting_object, predictor_file)


def evaluate_models(predictor_objects, test_dataset, evaluation_strategy="split", forecasting_horizon=1):
    models_error = []
    for predictor in predictor_objects:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,  # test dataset
            predictor=predictor_objects[predictor],  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        if evaluation_strategy == "split":
            agg_metrics, item_metrics = evaluator(iter(ts_it), iter(forecast_it), num_series=len(test_dataset))
        else:
            agg_metrics, item_metrics = backtest_metrics(
                test_dataset=test_dataset,
                train_dataset=test_dataset,
                predictor=predictor_objects[predictor],
                evaluator=evaluator,
                forecaster=predictor_objects[predictor]
            )
        agg_metrics.update({
            "predictor": predictor
        })
        models_error.append(agg_metrics)
    return models_error


def save_dataset(dataset_name, time_column_name, target_columns_names, model_folder, version_name):
    dataset = dataiku.Dataset(dataset_name)
    dataset_df = dataset.get_dataframe()
    virtual_fs = BytesIO()
    virtual_fs.seek(0)
    columns_to_save = []
    columns_to_save.append(time_column_name)
    columns_to_save.extend(target_columns_names)
    dataset_df.to_csv(virtual_fs, columns=columns_to_save)
    virtual_fs.seek(0)
    dataset_file_path = "{}/train_dataset.csv".format(version_name)
    model_folder.upload_stream(dataset_file_path, virtual_fs)
