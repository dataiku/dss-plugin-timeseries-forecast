import pandas as pd
import numpy as np
import os
import math
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
        timeseries_cross_validation (bool, optional): Whether to use timeseries cross-validation.
        rolling_windows_number (int, optional): Number of splits used in the training set. Mandatory for cross-validation.
        cutoff_period (int, optional): Number of time steps between each split. Mandatory for cross-validation.
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
        timeseries_cross_validation=False,
        rolling_windows_number=1,
        cutoff_period=-1,
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
            self.evaluation_forecasts_df = None
        self.metrics_df = None
        self.batch_size = batch_size
        self.user_num_batches_per_epoch = user_num_batches_per_epoch
        self.num_batches_per_epoch = None
        self.season_length = season_length
        self.mxnet_context = mxnet_context
        self.timeseries_cross_validation = timeseries_cross_validation
        self.rolling_windows_number = rolling_windows_number if timeseries_cross_validation else 1
        self.cutoff_period = cutoff_period

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

        self.rolling_windows_cut_lengths_train_test_pairs = self._compute_rolling_windows_cut_lengths_train_test_pairs()
        rolling_windows_unique_cut_lengths = self._compute_rolling_windows_unique_cut_lengths(
            self.rolling_windows_cut_lengths_train_test_pairs
        )

        self.gluon_list_datasets_by_cut_length = gluon_dataset.create_list_datasets(
            self.training_df, cut_lengths=rolling_windows_unique_cut_lengths
        )

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

    def train_evaluate_on_window(self, rolling_window_index, train_cut_length, test_cut_length, model, retrain=False):
        """Train and evaluate the model on the given rolling window.

        Args:
            rolling_window_index (int): Index of the rolling window
            train_cut_length (int): Cut length for the train dataset
            test_cut_length (int): Cut length for the test dataset
            model (Model): Model to train and evaluate
            retrain (bool, optional): If True, retrain the model on the last rolling window. Defaults to False.
        """

        train_list_dataset = self.gluon_list_datasets_by_cut_length[train_cut_length]
        test_list_dataset = self.gluon_list_datasets_by_cut_length[test_cut_length]

        last_window = rolling_window_index == len(self.rolling_windows_cut_lengths_train_test_pairs) - 1

        if self.timeseries_cross_validation:
            logger.info(
                f"Training and evaluating on rolling window #{rolling_window_index} - (train, test) cut lengths: ({train_cut_length}, {test_cut_length})"
            )

        item_metrics, forecasts_df = model.train_evaluate(
            train_list_dataset,
            test_list_dataset,
            make_forecasts=self.make_forecasts,
            retrain=retrain and last_window,
        )

        item_metrics.insert(0, METRICS_DATASET.ROLLING_WINDOWS, rolling_window_index)

        if self.make_forecasts:
            forecasts_df = forecasts_df.rename(columns={"index": self.time_column_name})

            forecasts_df[METRICS_DATASET.ROLLING_WINDOWS] = rolling_window_index

        return item_metrics, forecasts_df

    def train_evaluate(self, retrain=False):
        """
        Evaluate all the selected models (then retrain on complete data if specified),
        get the metrics dataframe and create the forecasts dataframe if make_forecasts=True.
        """
        metrics_df = pd.DataFrame()
        forecasts_df = pd.DataFrame()
        identifiers_columns = self.timeseries_identifiers_names or []

        for model in self.models:

            if self.make_forecasts:
                rolling_windows_forecasts_df = pd.DataFrame()

            for rolling_window_index, (train_cut_length, test_cut_length) in enumerate(
                self.rolling_windows_cut_lengths_train_test_pairs
            ):  # loop over the rolling windows
                item_metrics, single_window_forecasts_df = self.train_evaluate_on_window(
                    rolling_window_index, train_cut_length, test_cut_length, model, retrain
                )

                metrics_df = metrics_df.append(item_metrics)

                if self.make_forecasts:
                    rolling_windows_forecasts_df = rolling_windows_forecasts_df.append(single_window_forecasts_df)

            if self.make_forecasts:
                if forecasts_df.empty:
                    forecasts_df = rolling_windows_forecasts_df
                else:
                    forecasts_df = forecasts_df.merge(
                        rolling_windows_forecasts_df,
                        on=[self.time_column_name] + identifiers_columns + [METRICS_DATASET.ROLLING_WINDOWS],
                    )

        # aggregate metrics over rolling windows
        rolling_window_aggregations = self._aggregate_rolling_windows(metrics_df, identifiers_columns)
        metrics_df = metrics_df.append(rolling_window_aggregations)

        metrics_df[METRICS_DATASET.SESSION] = self.session_name
        self.metrics_df = self._reorder_metrics_df(metrics_df)

        if self.make_forecasts:
            evaluation_forecasts_df = self.training_df.merge(
                forecasts_df, on=[self.time_column_name] + identifiers_columns, how="left", indicator=True
            )

            self.evaluation_forecasts_df = self._prepare_evaluation_forecasts_df(
                evaluation_forecasts_df, identifiers_columns
            )

    def get_full_list_dataset(self):
        return self.gluon_list_datasets_by_cut_length[0]

    def get_shortest_list_dataset(self):
        return self.gluon_list_datasets_by_cut_length[
            self.prediction_length + (self.rolling_windows_number - 1) * self.cutoff_period
        ]

    def _prepare_evaluation_forecasts_df(self, evaluation_forecasts_df, identifiers_columns):
        """
        Add row origin (train or evaluation)
        Sort forecasts dataframe by timeseries identifiers (ascending), rolling window index (ascending) and time column (descending)
        Put rolling window index column at the end and add the session after it
        """

        evaluation_forecasts_df = add_row_origin(
            evaluation_forecasts_df, both=ROW_ORIGIN.EVALUATION, left_only=ROW_ORIGIN.TRAIN
        )

        evaluation_forecasts_df = evaluation_forecasts_df.sort_values(
            by=identifiers_columns + [METRICS_DATASET.ROLLING_WINDOWS] + [self.time_column_name],
            ascending=[True] * len(identifiers_columns) + [True, False],
        )

        rolling_window_column = evaluation_forecasts_df.pop(METRICS_DATASET.ROLLING_WINDOWS)
        evaluation_forecasts_df[METRICS_DATASET.ROLLING_WINDOWS] = rolling_window_column

        evaluation_forecasts_df[METRICS_DATASET.SESSION] = self.session_name

        return evaluation_forecasts_df

    def _aggregate_rolling_windows(self, metrics_df, identifiers_columns):
        """Aggregate metrics dataframe by rolling windows"""

        rolling_windows_aggregations_dict = {
            evaluation_metrics: "mean" for evaluation_metrics in EVALUATION_METRICS_DESCRIPTIONS
        }

        rolling_windows_aggregations_dict[METRICS_DATASET.TRAINING_TIME] = lambda x: x.sum() or np.nan
        rolling_windows_aggregations_dict[METRICS_DATASET.MODEL_PARAMETERS] = "min"
        rolling_windows_aggregations_dict[METRICS_DATASET.ROLLING_WINDOWS] = lambda x: METRICS_DATASET.AGGREGATED_ROW

        rolling_window_aggregations = metrics_df.groupby(
            [METRICS_DATASET.TARGET_COLUMN] + identifiers_columns + [METRICS_DATASET.MODEL_COLUMN],
            as_index=False,
        ).agg(rolling_windows_aggregations_dict)
        return rolling_window_aggregations

    def _reorder_metrics_df(self, metrics_df):
        """Sort rows by target column and put rolling windows aggregations and aggregated rows on top.

        Args:
            metrics_df (DataFrame): Dataframe of metrics of all timeseries.

        Returns:
            Ordered metrics DataFrame: First rolling windows aggregations, then aggregated rows, then the rest.
        """
        metrics_df = metrics_df.sort_values(by=[METRICS_DATASET.TARGET_COLUMN], ascending=True)
        ordered_metrics_list = [
            metrics_df[
                (metrics_df[METRICS_DATASET.ROLLING_WINDOWS] == METRICS_DATASET.AGGREGATED_ROW)
                & (metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW)
            ],
            metrics_df[
                (metrics_df[METRICS_DATASET.ROLLING_WINDOWS] == METRICS_DATASET.AGGREGATED_ROW)
                & (metrics_df[METRICS_DATASET.TARGET_COLUMN] != METRICS_DATASET.AGGREGATED_ROW)
            ],
            metrics_df[
                (metrics_df[METRICS_DATASET.ROLLING_WINDOWS] != METRICS_DATASET.AGGREGATED_ROW)
                & (metrics_df[METRICS_DATASET.TARGET_COLUMN] != METRICS_DATASET.AGGREGATED_ROW)
            ],
        ]

        return pd.concat(ordered_metrics_list, axis=0).reset_index(drop=True)

    def get_evaluation_forecasts_to_display(self):
        evaluation_forecasts_df = self.evaluation_forecasts_df.copy()
        if not self.timeseries_cross_validation:
            evaluation_forecasts_df = evaluation_forecasts_df.drop(columns=[METRICS_DATASET.ROLLING_WINDOWS])
        return evaluation_forecasts_df

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

    def get_evaluation_metrics_to_display(self):
        """
        Replace __aggregated__ by target column name and remove other rows when only one target
        and no timeseries identifiers.
        If no timeseries cross-validation, remove rows other than the windows aggregation rows,
        and drop rolling window indew column.

        Returns:
            Dataframe of metrics to display to users.
        """
        evaluation_metrics_df = self.metrics_df.copy()
        evaluation_metrics_df.columns = [
            column.lower() if column in EVALUATION_METRICS_DESCRIPTIONS else column
            for column in evaluation_metrics_df.columns
        ]
        if len(self.target_columns_names) == 1 and not self.timeseries_identifiers_names:
            evaluation_metrics_df = evaluation_metrics_df[
                evaluation_metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW
            ]
            evaluation_metrics_df[METRICS_DATASET.TARGET_COLUMN] = self.target_columns_names[0]
        if not self.timeseries_cross_validation:
            evaluation_metrics_df = evaluation_metrics_df[
                evaluation_metrics_df[METRICS_DATASET.ROLLING_WINDOWS] == METRICS_DATASET.AGGREGATED_ROW
            ]
            evaluation_metrics_df = evaluation_metrics_df.drop(columns=[METRICS_DATASET.ROLLING_WINDOWS])

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
        for timeseries in self.get_shortest_list_dataset().list_data:
            timeseries_length = len(timeseries[TIMESERIES_KEYS.TARGET])
            num_samples = math.ceil(timeseries_length / self.prediction_length)
            num_samples_total += num_samples
        optimal_num_batches_per_epoch = max(math.ceil(num_samples_total / self.batch_size), 50)
        logger.info(
            f"Number of batches per epoch automatically scaled to training data size: {optimal_num_batches_per_epoch}"
        )
        return optimal_num_batches_per_epoch

    def _compute_rolling_windows_cut_lengths_train_test_pairs(self):
        """Compute the rolling windows cut lengths for each train/test pair of the cross-validation.
        No cross-validation is equivalent to 1 rolling window with a cut length of prediction_length for the train set and 0 for the test set.

        Example:
        The timeseries [1, 2, 3, 4, 5, 6] with a cut_length of 2 becomes [1, 2, 3, 4].
        With a prediction_length of 2, a cutoff_period of 1, and 3 rolling windows, the cut lengths train test pairs are:
        [(4, 2), (3, 1), (2, 0)]
        This creates the following rolling windows:
        1. cut lengths pair (4, 2) => train: [1, 2], test: [1, 2, 3, 4]
        2. cut lengths pair (3, 1) => train: [1, 2, 3], test: [1, 2, 3, 4, 5]
        3. cut lengths pair (2, 0) => train: [1, 2, 3, 4], test: [1, 2, 3, 4, 5, 6]
        """
        self.cutoff_period = self.prediction_length // 2 if self.cutoff_period == -1 else self.cutoff_period
        if self.timeseries_cross_validation:
            return [
                (self.prediction_length + (i - 1) * self.cutoff_period, (i - 1) * self.cutoff_period)
                for i in range(self.rolling_windows_number, 0, -1)
            ]
        else:
            return [(self.prediction_length, 0)]

    def _compute_rolling_windows_unique_cut_lengths(self, rolling_windows_cut_lengths_pairs):
        """Compute unique rolling windows cut lengths from the list of train/test cut lengths pairs.

        Args:
            rolling_windows_cut_lengths_pairs (list): List of tuples (cut_length_train, cut_length_test)

        Returns:
            list: List of unique cut lengths

        Example:
        [(4, 2), (3, 1), (2, 0)] generates [4, 3, 2, 1, 0]
        """
        return list(set([cut_length for pair in rolling_windows_cut_lengths_pairs for cut_length in pair]))
