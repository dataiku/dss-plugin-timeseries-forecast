
import re
import io
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.npts import NPTSEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.trainer import Trainer
# from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.naive_2 import Naive2Predictor
import os
import dill as pickle
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import backtest_metrics
import pandas as pd
import json
import gzip
import logging

AVAILABLE_MODELS = [
    "naive", "simplefeedforward", "deepfactor", "deepar", "lstnet", "nbeats",
    "npts", "transformer"
]

EVALUATION_METRICS = [
    "MSE", "MASE", "MAPE", "sMAPE", "MSIS", "RMSE", "ND", "mean_wQuantileLoss"
]

class METRICS_DATASET:
    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"

ESTIMATOR = 'estimator'
CAN_USE_EXTERNAL_FEATURES = 'can_use_external_feature' # TODO implement
TRAINER = 'trainer'

MODEL_DESCRIPTORS = {
    "default": {},
    "deepar": {
        CAN_USE_EXTERNAL_FEATURES: True,
        ESTIMATOR: DeepAREstimator,
        TRAINER: Trainer
    },
    "deepfactor": {
        ESTIMATOR: DeepFactorEstimator,
        TRAINER: Trainer
    },
    "lstnet": {
        ESTIMATOR: LSTNetEstimator,
        TRAINER: Trainer
    },
    "naive": {
        ESTIMATOR: Naive2Predictor,
        TRAINER: None
    },
    "nbeats": {
        ESTIMATOR: NBEATSEstimator,
        TRAINER: Trainer
    },
    "npts": {
        ESTIMATOR: NPTSEstimator,
        TRAINER: None
    },
    "simplefeedforward": {
        CAN_USE_EXTERNAL_FEATURES: False,
        ESTIMATOR: SimpleFeedForwardEstimator,
        TRAINER: Trainer
    },
    "transformer": {
        ESTIMATOR: TransformerEstimator,
        TRAINER: Trainer
    }
}


def read_from_folder(folder, path, obj_type):
    logging.info("Timeseries forecast - Loading {}".format(os.path.join(folder.get_path(), path)))
    with folder.get_download_stream(path) as stream:
        if obj_type == 'pickle':
            return pickle.loads(stream.read())
        if obj_type == 'pickle.gz':
            with gzip.GzipFile(fileobj=stream) as fgzip:
                return pickle.loads(fgzip.read())
        elif obj_type == 'csv':
            data = io.StringIO(stream.read().decode())
            return pd.read_csv(data)
        elif obj_type == 'csv.gz':
            with gzip.GzipFile(fileobj=stream) as fgzip:
                return pd.read_csv(fgzip)
        else:
            raise ValueError("Can only read objects of type ['pickle', 'pickle.gz', 'csv', 'csv.gz'] from folder, not '{}'".format(obj_type))


def write_to_folder(obj, folder, path, obj_type):
    logging.info("Timeseries forecast - Saving {}".format(os.path.join(folder.get_path(), path)))
    with folder.get_writer(path) as writer:
        if obj_type == 'pickle':
            writeable = pickle.dumps(obj)
            writer.write(writeable)
        elif obj_type == 'pickle.gz':
            writeable = pickle.dumps(obj)
            with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=9) as fgzip:
                fgzip.write(writeable)
        elif obj_type == 'json':
            writeable = json.dumps(obj).encode()
            writer.write(writeable)
        elif obj_type == 'csv':
            writeable = obj.to_csv(sep=',', na_rep='', header=True, index=False).encode()
            writer.write(writeable)
        elif obj_type == 'csv.gz':
            with gzip.GzipFile(fileobj=writer, mode='wb') as fgzip:
                fgzip.write(obj.to_csv(index=False).encode())
        else:
            raise ValueError("Can only write objects of type ['pickle', 'pickle.gz', 'json', 'csv', 'csv.gz'] to folder, not '{}'".format(obj_type))


def get_models_parameters(config):
    models_parameters = {}
    for model in AVAILABLE_MODELS:
        if is_activated(config, model):
            model_presets = get_model_presets(config, model)
            if 'prediction_length' in model_presets.get('kwargs', {}):
                raise ValueError("The value for 'prediction_length' cannot be changed")
            models_parameters.update({
                model: model_presets
            })
    return models_parameters


def is_activated(config, model):
    return config.get("{}_model_activated".format(model), False)


def get_model_kwargs(config, model):
    return config.get("{}_model_kwargs".format(model))


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


def get_model_descriptor(model):
    model_descriptor = MODEL_DESCRIPTORS.get(model)
    if model_descriptor is None:
        return MODEL_DESCRIPTORS.get('default')
    else:
        return model_descriptor


def get_estimator(model, model_parameters, **kwargs):
    kwargs.update(model_parameters.get("kwargs", {}))
    model_descriptor = get_model_descriptor(model)
    estimator = model_descriptor.get(ESTIMATOR)
    return None if estimator is None else estimator(**kwargs)


def get_trainer(model, **kwargs):
    model_descriptor = get_model_descriptor(model)
    trainer = model_descriptor.get(TRAINER)
    return None if trainer is None else trainer(**kwargs)


def can_model_use_external_feature(model):
    model_descriptor = get_model_descriptor(model)
    return model_descriptor.get(CAN_USE_EXTERNAL_FEATURES, False)


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


def set_column_description(output_dataset, column_description_dict, input_dataset=None):
    """Set column descriptions of the output dataset based on a dictionary of column descriptions

    Retains the column descriptions from the input dataset if the column name matches.

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_description_dict: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions
    """
    output_dataset_schema = output_dataset.read_schema()
    input_dataset_schema = []
    input_columns_names = []
    if input_dataset is not None:
        input_dataset_schema = input_dataset.read_schema()
        input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_description_dict.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)


def assert_continuous_time_column(dataframe, time_column_name, time_granularity_unit, time_granularity_step):
    """ raise an explicit error message """
    is_continuous = check_continuous_time_column(dataframe, time_column_name, time_granularity_unit, time_granularity_step)
    if not is_continuous:
        frequency = "{}{}".format(time_granularity_step, time_granularity_unit)
        error_message = f"Time column {time_column_name} doesn't have regular time intervals of frequency {frequency}."
        if time_granularity_unit in ['M', 'Y']:
            unit_name = 'Month' if time_granularity_step == 'M' else 'Year'
            error_message += f"For {unit_name} frequency, timestamps must be end of {unit_name} (for e.g. '2020-12-31 00:00:00')"
        raise ValueError(error_message)


def check_continuous_time_column(dataframe, time_column_name, time_granularity_unit, time_granularity_step):
    """ check that all timesteps are identical and follow the chosen frequency """
    dataframe[time_column_name] = pd.to_datetime(dataframe[time_column_name]).dt.tz_localize(tz=None)
    frequency = "{}{}".format(time_granularity_step, time_granularity_unit)

    start_date = dataframe[time_column_name].iloc[0]
    end_date = dataframe[time_column_name].iloc[-1]

    date_range_df = pd.date_range(start=start_date, end=end_date, freq=frequency).to_frame(index=False)

    if len(date_range_df.index) != len(dataframe.index) or not dataframe[time_column_name].equals(date_range_df[0]):
        return False
    return True
