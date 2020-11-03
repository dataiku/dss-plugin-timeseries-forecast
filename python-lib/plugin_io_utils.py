
import re
import io
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
    "transformer"
]

EVALUATION_METRICS = [
    "MSE", "MASE", "MAPE", "sMAPE", "MSIS", "RMSE", "ND", "mean_wQuantileLoss"
]


class METRICS_DATASET:
    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"


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


def remove_timezone_information(dataframe, time_column_name):
    dataframe[time_column_name] = pd.to_datetime(dataframe[time_column_name]).dt.tz_localize(tz=None)


def check_external_features_future_dataset_schema(gluon_train_dataset, external_features_future_dataset):
    """
    check that schema of external_features_future_dataset contains exactly and only
    time_column_name | feat_dynamic_real_columns_names | identifiers.keys()
    """
    external_features_future_columns = [column['name'] for column in external_features_future_dataset.read_schema()]
    train_data_sample = gluon_train_dataset.list_data[0]
    expected_columns = [train_data_sample['time_column_name']] + train_data_sample['feat_dynamic_real_columns_names']
    if 'identifiers' in train_data_sample:
        expected_columns += list(train_data_sample['identifiers'].keys())
    if set(external_features_future_columns) != set(expected_columns):
        raise ValueError("The dataset of future values of external features must contains exactly the following columns: {}".format(expected_columns))
