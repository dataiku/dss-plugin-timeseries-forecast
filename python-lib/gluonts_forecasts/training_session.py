import pandas as pd
import os
import math
from pandas.api.types import is_numeric_dtype, is_string_dtype
from gluonts_forecasts.model import Model
from dku_constants import (
    METRICS_DATASET,
    METRICS_COLUMNS_DESCRIPTIONS,
    TIMESERIES_KEYS,
    EVALUATION_METRICS_DESCRIPTIONS,
    ROW_ORIGIN,
)
from gluonts_forecasts.gluon_dataset import DkuGluonDataset
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from gluonts_forecasts.utils import add_row_origin
from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


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
        training_df (DataFrame): Training dataframe
        make_forecasts (bool): True to output the evaluation predictions of the last prediction_length time steps
        external_features_columns_names (list, optional): List of columns with dynamic real features over time
        timeseries_identifiers_names (list, optional): Columns to identify multiple time series when data is in long format
        batch_size (int, optional): Size of batch used by the GluonTS Trainer class
        user_num_batches_per_epoch (int, optional): Number of batches per epoch selected by user. -1 means to compute scaled number
        num_batches_per_epoch (int, optional): Number of batches per epoch
        season_length (int, optional): Length of the seasonality parameter.
        mxnet_context (mxnet.context.Context, optional): MXNet context to use for Deep Learning models training.
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
        user_num_batches_per_epoch=None,
        season_length=None,
        mxnet_context=None,
    ):
        self.models_parameters = models_parameters
        self.models = []
        self.glutonts_dataset = None
        self.training_df = training_df
        self.prediction_length = prediction_length
        self.target_columns_names = target_columns_names
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.epoch = epoch
        self.make_forecasts = make_forecasts
        self.external_features_columns_names = external_features_columns_names
        self.use_external_features = bool(external_features_columns_names)
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.session_name = None
        self.session_path = None
        if self.make_forecasts:
            self.forecasts_df = pd.DataFrame()
            self.evaluation_forecasts_df = None
        self.evaluation_train_list_dataset = None
        self.full_list_dataset = None
        self.metrics_df = None
        self.batch_size = batch_size
        self.user_num_batches_per_epoch = user_num_batches_per_epoch
        self.num_batches_per_epoch = None
        self.season_length = season_length
        self.mxnet_context = mxnet_context

    def init(self, session_name, partition_root=None):
        """Create the session_path.

        Args:
            session_name (Timestamp)
            partition_root (str, optional): Partition root path, concatenated to session_name to create the session_path. Defaults to None.
        """
        self.session_name = session_name
        if partition_root is None:
            self.session_path = session_name
        else:
            self.session_path = os.path.join(partition_root, session_name)

    def create_gluon_list_datasets(self):
        """Create train and test gluon list datasets.
        The last prediction_length time steps are removed from each timeseries of the train dataset.
        Compute optimal num_batches_per_epoch value based on the train dataset size.
        """

        gluon_dataset = DkuGluonDataset(
            time_column_name=self.time_column_name,
            frequency=self.frequency,
            target_columns_names=self.target_columns_names,
            timeseries_identifiers_names=self.timeseries_identifiers_names,
            external_features_columns_names=self.external_features_columns_names,
            min_length=2 * self.prediction_length,  # Assuming that context_length = prediction_length
        )

        gluon_list_datasets = gluon_dataset.create_list_datasets(
            self.training_df, cut_lengths=[self.prediction_length, 0]
        )
        self.evaluation_train_list_dataset = gluon_list_datasets[0]
        self.full_list_dataset = gluon_list_datasets[1]

        if self.user_num_batches_per_epoch == -1:
            self.num_batches_per_epoch = self._compute_optimal_num_batches_per_epoch()
        else:
            self.num_batches_per_epoch = self.user_num_batches_per_epoch

    def instantiate_models(self):
        """Instantiate all the selected models."""
        for model_name, model_parameters in self.models_parameters.items():
            self.models.append(
                Model(
                    model_name,
                    model_parameters=model_parameters,
                    frequency=self.frequency,
                    prediction_length=self.prediction_length,
                    epoch=self.epoch,
                    use_external_features=self.use_external_features,
                    batch_size=self.batch_size,
                    num_batches_per_epoch=self.num_batches_per_epoch,
                    season_length=self.season_length,
                    mxnet_context=self.mxnet_context,
                )
            )

    def train_evaluate(self, retrain=False):
        """Call the right train and evaluate function depending on the need to make forecasts."""

        if self.make_forecasts:
            self._train_evaluate_make_forecast(retrain)
        else:
            self._train_evaluate(retrain)

    def _train_evaluate(self, retrain):
        """Evaluate all the selected models (then retrain on complete data if specified) and get the metrics dataframe."""
        metrics_df = pd.DataFrame()
        for model in self.models:
            item_metrics = model.train_evaluate(
                self.evaluation_train_list_dataset, self.full_list_dataset, retrain=retrain
            )[0]
            metrics_df = metrics_df.append(item_metrics)
        metrics_df[METRICS_DATASET.SESSION] = self.session_name
        self.metrics_df = self._reorder_metrics_df(metrics_df)

    def _train_evaluate_make_forecast(self, retrain):
        """Evaluate all the selected models (then retrain on complete data if specified), get the metrics dataframe and create the forecasts dataframe."""
        metrics_df = pd.DataFrame()
        for model in self.models:
            (item_metrics, identifiers_columns, forecasts_df) = model.train_evaluate(
                self.evaluation_train_list_dataset, self.full_list_dataset, make_forecasts=True, retrain=retrain
            )
            forecasts_df = forecasts_df.rename(columns={"index": self.time_column_name})
            if self.forecasts_df.empty:
                self.forecasts_df = forecasts_df
            else:
                self.forecasts_df = self.forecasts_df.merge(
                    forecasts_df, on=[self.time_column_name] + identifiers_columns
                )
            metrics_df = metrics_df.append(item_metrics)
        metrics_df[METRICS_DATASET.SESSION] = self.session_name
        self.metrics_df = self._reorder_metrics_df(metrics_df)

        self.evaluation_forecasts_df = self.training_df.merge(
            self.forecasts_df, on=[self.time_column_name] + identifiers_columns, how="left", indicator=True
        )

        self.evaluation_forecasts_df = add_row_origin(
            self.evaluation_forecasts_df, both=ROW_ORIGIN.EVALUATION, left_only=ROW_ORIGIN.TRAIN
        )

        # sort forecasts dataframe by timeseries identifiers (ascending) and time column (descending)
        self.evaluation_forecasts_df = self.evaluation_forecasts_df.sort_values(
            by=identifiers_columns + [self.time_column_name], ascending=[True] * len(identifiers_columns) + [False]
        )
        self.evaluation_forecasts_df[METRICS_DATASET.SESSION] = self.session_name

    def _reorder_metrics_df(self, metrics_df):
        """Sort rows by target column and put aggregated rows on top.

        Args:
            metrics_df (DataFrame): Dataframe of metrics of all timeseries.

        Returns:
            Ordered metrics DataFrame.
        """
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
        """Explain the meaning of the forecasts columns.

        Returns:
            Dictionary of description (value) by column (key).
        """
        column_descriptions = METRICS_COLUMNS_DESCRIPTIONS.copy()
        available_models = ModelConfigRegistry().list_available_models()
        for column in self.evaluation_forecasts_df.columns:
            model = next((model for model in available_models if model in column), None)
            if model:
                column_split = column.split(f"{model}_")
                if len(column_split) > 1:
                    target_name = [1]
                    column_descriptions[column] = f"Median forecasts of {target_name} using {model} model"
        return column_descriptions

    def get_metrics_df(self):
        return self.metrics_df

    def get_evaluation_metrics_df(self):
        """Replace __aggregated__ by target column name and remove other rows when only one target
        and no timeseries identifiers.

        Returns:
            Dataframe of metrics to display to users.
        """
        evaluation_metrics_df = self.metrics_df.copy()
        evaluation_metrics_df.columns = [
            column.lower() if column in EVALUATION_METRICS_DESCRIPTIONS else column
            for column in evaluation_metrics_df.columns
        ]
        if len(self.target_columns_names) == 1 and not self.timeseries_identifiers_names:
            evaluation_metrics_df = self.metrics_df.copy()
            evaluation_metrics_df = evaluation_metrics_df[
                evaluation_metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW
            ]
            evaluation_metrics_df[METRICS_DATASET.TARGET_COLUMN] = self.target_columns_names[0]
        return evaluation_metrics_df

    def create_evaluation_results_columns_descriptions(self):
        """Explain the meaning of the metrics dataset columns.

        Returns:
            Dictionary of description (value) by column (key).
        """
        column_descriptions = METRICS_COLUMNS_DESCRIPTIONS.copy()
        for column in EVALUATION_METRICS_DESCRIPTIONS:
            column_descriptions[column.lower()] = EVALUATION_METRICS_DESCRIPTIONS[column]
        return column_descriptions

    def _compute_optimal_num_batches_per_epoch(self):
        """Compute the optimal value of num batches per epoch to scale to the training data size.
        With this formula, each timestep will on average be in 2 samples, once in the context part and once in the prediction part.
        """
        num_samples_total = 0
        for timeseries in self.evaluation_train_list_dataset.list_data:
            timeseries_length = len(timeseries[TIMESERIES_KEYS.TARGET])
            num_samples = math.ceil(timeseries_length / self.prediction_length)
            num_samples_total += num_samples
        optimal_num_batches_per_epoch = max(math.ceil(num_samples_total / self.batch_size), 50)
        logger.info(
            f"Number of batches per epoch automatically scaled to training data size: {optimal_num_batches_per_epoch}"
        )
        return optimal_num_batches_per_epoch
