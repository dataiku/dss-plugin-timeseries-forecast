import pandas as pd
import os
from pandas.api.types import is_numeric_dtype, is_string_dtype
from gluonts_forecasts.model import Model
from constants import METRICS_DATASET, MODEL_LABELS
from gluonts_forecasts.gluon_dataset import GluonDataset


class TrainingSession:
    """
    Class to train and evaluate multiple GluonTS estimators on a training dataframe, and to retrieve an aggregated metrics dataframe

    Attributes:
        target_columns_names (list): List of column names to predict
        time_column_name (str)
        frequency (str): Pandas timeseries frequency (e.g. '3M')
        epoch (int): Number of epochs used by the GluonTS Trainer class
        models_parameters (dict): Dictionary of model names (key) and their parameters (value)
        prediction_length (int): Number of time steps to predict
        training_df (Pandas.DataFrame): Training dataframe
        make_forecasts (bool): True to output the evaluation predictions of the last prediction_length time steps
        external_features_columns_names (list): List of columns with dynamic real features over time
        timeseries_identifiers_names (list): Columns to identify multiple time series when data is in long format
        batch_size (int): Size of batch used by the GluonTS Trainer class
        gpu (str): Not implemented
        context_length (int): Number of time steps used by model to make predictions
    """

    def __init__(
        self,
        target_columns_names,
        time_column_name,
        frequency,
        epoch,
        models_parameters,
        prediction_length,
        training_df,
        make_forecasts,
        external_features_columns_names=None,
        timeseries_identifiers_names=None,
        batch_size=None,
        gpu=None,
        context_length=None,
    ):
        self.models_parameters = models_parameters
        self.model_names = []
        self.models = None
        self.glutonts_dataset = None
        self.training_df = training_df
        self.prediction_length = prediction_length
        self.target_columns_names = target_columns_names
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.epoch = epoch
        self.make_forecasts = make_forecasts
        self.external_features_columns_names = external_features_columns_names
        self.use_external_features = len(external_features_columns_names) > 0
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.session_name = None
        self.session_path = None
        if self.make_forecasts:
            self.forecasts_df = pd.DataFrame()
            self.evaluation_forecasts_df = None
        self.train_list_dataset = None
        self.test_list_dataset = None
        self.metrics_df = None
        self.batch_size = batch_size
        self.gpu = gpu
        self.context_length = context_length

    def init(self, session_name, partition_root=None):
        self.session_name = session_name
        if partition_root is None:
            self.session_path = session_name
        else:
            self.session_path = os.path.join(partition_root, session_name)
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                Model(
                    model_name,
                    model_parameters=model_parameters,
                    frequency=self.frequency,
                    prediction_length=self.prediction_length,
                    epoch=self.epoch,
                    use_external_features=self.use_external_features,
                    batch_size=self.batch_size,
                    gpu=self.gpu,
                    context_length=self.context_length,
                )
            )
        try:
            self.training_df[self.time_column_name] = pd.to_datetime(self.training_df[self.time_column_name]).dt.tz_localize(tz=None)
        except Exception:
            raise ValueError("Time column '{}' cannot be parsed as date.".format(self.time_column_name))

        self._check_target_columns_types()
        self._check_external_features_columns_types()
        self._check_timeseries_identifiers_columns_types()

    def train(self):
        """ Train all the selected models on all data """
        for model in self.models:
            model.train(self.test_list_dataset)

    def evaluate(self, evaluation_strategy):
        """
        Train all the selected models on all but the last prediction_length time steps,
        and evaluate on all data according to the selected evaluation_strategy
        """
        gluon_dataset = GluonDataset(
            dataframe=self.training_df,
            time_column_name=self.time_column_name,
            frequency=self.frequency,
            target_columns_names=self.target_columns_names,
            timeseries_identifiers_names=self.timeseries_identifiers_names,
            external_features_columns_names=self.external_features_columns_names,
        )

        if evaluation_strategy == "split":
            self.train_list_dataset = gluon_dataset.create_list_dataset(cut_length=self.prediction_length)
            self.test_list_dataset = gluon_dataset.create_list_dataset()
        else:
            raise Exception("{} evaluation strategy not implemented".format(evaluation_strategy))
        self.metrics_df = self._compute_all_evaluation_metrics()

    def _compute_all_evaluation_metrics(self):
        """ Evaluate all the selected models, output the metrics dataframe and create the forecasts dataframe if make_forecasts is True """
        metrics_df = pd.DataFrame()
        for model in self.models:
            if self.make_forecasts:
                (item_metrics, identifiers_columns, forecasts_df) = model.evaluate(self.train_list_dataset, self.test_list_dataset, make_forecasts=True)
                forecasts_df = forecasts_df.rename(columns={"index": self.time_column_name})
                if self.forecasts_df.empty:
                    self.forecasts_df = forecasts_df
                else:
                    self.forecasts_df = self.forecasts_df.merge(forecasts_df, on=[self.time_column_name] + identifiers_columns)
            else:
                (item_metrics, identifiers_columns) = model.evaluate(self.train_list_dataset, self.test_list_dataset)
            metrics_df = metrics_df.append(item_metrics)
        metrics_df[METRICS_DATASET.SESSION] = self.session_name
        orderd_metrics_df = self._reorder_metrics_df(metrics_df)

        if self.make_forecasts:
            self.evaluation_forecasts_df = self.training_df.merge(
                self.forecasts_df,
                on=[self.time_column_name] + identifiers_columns,
                how="left",
            )
            self.evaluation_forecasts_df[METRICS_DATASET.SESSION] = self.session_name

        return orderd_metrics_df

    def _reorder_metrics_df(self, metrics_df):
        """ sort rows by target column and put aggregated rows on top """
        metrics_df = metrics_df.sort_values(by=[METRICS_DATASET.TARGET_COLUMN], ascending=True)
        orderd_metrics_df = pd.concat(
            [
                metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW],
                metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN] != METRICS_DATASET.AGGREGATED_ROW],
            ],
            axis=0,
        ).reset_index(drop=True)
        return orderd_metrics_df

    def get_evaluation_forecasts_df(self):
        return self.evaluation_forecasts_df

    def create_evaluation_forecasts_column_description(self):
        """ explain the meaning of the forecasts columns """
        column_descriptions = {}
        for column in self.evaluation_forecasts_df.columns:
            suffix = column.split("_")[-1]
            if suffix in MODEL_LABELS.values():
                column_descriptions[column] = "Mean forecasts of {} using {} model".format(column.split("_{}".format(suffix))[0], suffix)
        return column_descriptions

    def get_metrics_df(self):
        return self.metrics_df

    def _check_target_columns_types(self):
        for column_name in self.target_columns_names:
            if not is_numeric_dtype(self.training_df[column_name]):
                raise ValueError("Target column '{}' must be of numerical data type.".format(column_name))

    def _check_external_features_columns_types(self):
        for column_name in self.external_features_columns_names:
            if not is_numeric_dtype(self.training_df[column_name]):
                raise ValueError("External feature column '{}' must be of numerical data type.".format(column_name))

    def _check_timeseries_identifiers_columns_types(self):
        for column_name in self.timeseries_identifiers_names:
            if not is_numeric_dtype(self.training_df[column_name]) and not is_string_dtype(self.training_df[column_name]):
                raise ValueError("Timeseries identifiers column '{}' must be of numerical or string data type.".format(column_name))
