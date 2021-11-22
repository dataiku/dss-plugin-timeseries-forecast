from dku_constants import (
    EVALUATION_METRICS_DESCRIPTIONS,
    METRICS_DATASET,
    TIMESERIES_KEYS,
    CUSTOMISABLE_FREQUENCIES_OFFSETS,
)
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts_forecasts.gluon_dataset import remove_unused_external_features

from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from gluonts_forecasts.utils import concat_timeseries_per_identifiers, concat_all_timeseries, quantile_forecasts_series
from time import perf_counter
from pandas.tseries.frequencies import to_offset
from safe_logger import SafeLogger
import json
import multiprocessing


logger = SafeLogger("Forecast plugin")


class ModelTrainingError(Exception):
    """Custom exception raised when the model training fails"""

    pass


class ModelPredictionError(Exception):
    """Custom exception raised when the model predictions fail"""

    pass


class Model:
    """
    Wrapper class to train and evaluate a GluonTS estimator, and retrieve the evaluation metrics and predictions

    Attributes:
        model_name (str): Model name of a ModelConfig()
        model_parameters (dict): Kwargs of model parameters
        custom_frequency (str): Pandas timeseries frequency not necessarily supported by GluonTS Estimators (e.g. 'W-MON')
        frequency (str): Pandas timeseries frequency (e.g. '3M') supported by GluonTS Estimators
        prediction_length (int): Number of time steps to predict
        epoch (int): Number of epochs used by the GluonTS Trainer class
        use_external_features (bool): If the model will be fed external features (use_feat_dynamic_real in GluonTS)
        use_seasonality (bool): If the model will be fed a seasonality parameter
        batch_size (int): Size of batch used by the GluonTS Trainer class
        mxnet_context (mxnet.context.Context): MXNet context to use for Deep Learning models training.
    """

    def __init__(
        self,
        model_name,
        model_parameters,
        frequency,
        prediction_length,
        epoch,
        use_external_features=False,
        batch_size=None,
        num_batches_per_epoch=None,
        season_length=None,
        mxnet_context=None,
    ):
        self.model_config = ModelConfigRegistry().get_model(model_name)

        self.model_name = model_name
        self.model_parameters = model_parameters
        self.custom_frequency = frequency
        self.frequency = (
            frequency
            if not isinstance(to_offset(frequency), CUSTOMISABLE_FREQUENCIES_OFFSETS)
            else to_offset(frequency).name.split("-")[0]
        )
        self.prediction_length = prediction_length
        self.epoch = epoch
        self.use_external_features = use_external_features and self.model_config.can_use_external_feature()
        self.use_seasonality = self.model_config.can_use_seasonality()
        self.use_batch_size = self.model_config.can_use_batch_size()
        self.mxnet_context = mxnet_context

        self.estimator_kwargs = {
            "freq": self.frequency,
            "prediction_length": self.prediction_length,
        }
        trainer_kwargs = {"ctx": self.mxnet_context, "epochs": self.epoch}

        self.num_batches_per_epoch = num_batches_per_epoch
        if self.num_batches_per_epoch is not None:
            trainer_kwargs.update({"num_batches_per_epoch": self.num_batches_per_epoch})
        self.trainer = self.model_config.trainer(**trainer_kwargs)
        if self.trainer is not None:
            self.estimator_kwargs.update({"trainer": self.trainer})
        else:
            self.mxnet_context = None

        self.batch_size = batch_size
        if self.use_batch_size and self.batch_size is not None:
            self.estimator_kwargs.update({"batch_size": self.batch_size})

        self.season_length = season_length
        if self.use_seasonality and self.season_length is not None:
            self.estimator_kwargs.update({"season_length": self.season_length})

        if self.use_external_features:
            self.estimator_kwargs.update({"use_feat_dynamic_real": True})
        self.estimator = self.model_config.estimator(self.model_parameters, **self.estimator_kwargs)
        self.predictor = None
        self.evaluation_time = 0
        self.retraining_time = 0

    def get_name(self):
        return self.model_name

    def train(self, train_list_dataset, reinit=True):
        """Train model on train_list_dataset and re-instanciate estimator if reinit=True"""
        start = perf_counter()
        logger.info(f"Re-training {self.model_config.get_label()} model on entire dataset ...")

        if reinit:  # re-instanciate model to re-initialize model parameters
            self.estimator = self.model_config.estimator(self.model_parameters, **self.estimator_kwargs)

        self.predictor = self._train_estimator(train_list_dataset)

        self.retraining_time = perf_counter() - start
        logger.info(
            f"Re-training {self.model_config.get_label()} model on entire dataset: Done in {self.retraining_time:.2f} seconds"
        )

    def train_evaluate(self, train_list_dataset, test_list_dataset, make_forecasts=False, retrain=False):
        """Train Model on train_list_dataset and evaluate it on test_list_dataset. Then retrain on test_list_dataset if retrain=True.

        Args:
            train_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.
            test_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.
            make_forecasts (bool, optional): Whether to make the evaluation forecasts and return them. Defaults to False.
            retrain (bool, optional): Whether to retrain model on test_list_dataset after the evaluation. Defaults to False.

        Returns:
            Evaluation metrics DataFrame for each target and aggregated.
            List of timeseries identifiers column names. Empty list if none found in train_list_dataset.
            DataFrame of predictions for the last prediction_length timesteps of the test_list_dataset timeseries if make_forecasts is True.
        """
        if not self.use_external_features and TIMESERIES_KEYS.FEAT_DYNAMIC_REAL in train_list_dataset.list_data[0]:
            train_list_dataset = remove_unused_external_features(train_list_dataset, self.frequency)
            test_list_dataset = remove_unused_external_features(test_list_dataset, self.frequency)

        logger.info(f"Evaluating {self.model_config.get_label()} model performance...")
        start = perf_counter()
        evaluation_predictor = self._train_estimator(train_list_dataset)

        agg_metrics, item_metrics, forecasts = self._make_evaluation_predictions(
            evaluation_predictor, test_list_dataset
        )
        self.evaluation_time = perf_counter() - start
        logger.info(
            f"Evaluating {self.model_config.get_label()} model performance: Done in {self.evaluation_time:.2f} seconds"
        )

        if retrain:
            self.train(test_list_dataset)

        metrics, identifiers_columns = self._format_metrics(agg_metrics, item_metrics, train_list_dataset)

        if make_forecasts:
            median_forecasts_timeseries = self._compute_median_forecasts_timeseries(forecasts, train_list_dataset)
            multiple_df = concat_timeseries_per_identifiers(median_forecasts_timeseries)
            forecasts_df = concat_all_timeseries(multiple_df)
            return metrics, identifiers_columns, forecasts_df

        return metrics, identifiers_columns

    def _make_evaluation_predictions(self, predictor, test_list_dataset):
        """Evaluate predictor and generate sample forecasts.

        Args:
            predictor (gluonts.model.predictor.Predictor): Trained object used to make forecasts.
            test_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.

        Returns:
            Dictionary of aggregated metrics over all timeseries.
            DataFrame of metrics for each timeseries (i.e., each target column).
            List of gluonts.model.forecast.Forecast (objects storing the predicted distributions as samples).
        """
        try:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_list_dataset, predictor=predictor, num_samples=100
            )
            forecasts = list(forecast_it)
        except Exception as err:
            raise ModelPredictionError(
                f"GluonTS '{self.model_name}' model crashed when making predictions. Full error: {err}"
            )
        evaluator = Evaluator(num_workers=min(2, multiprocessing.cpu_count()))
        agg_metrics, item_metrics = evaluator(ts_it, forecasts, num_series=len(test_list_dataset))
        return agg_metrics, item_metrics, forecasts

    def _format_metrics(self, agg_metrics, item_metrics, train_list_dataset):
        """Append agg_metrics to item_metrics and add new columns: model_name, target_column, identifiers_columns

        Args:
            agg_metrics (dict): Dictionary of aggregated metrics over all timeseries.
            item_metrics (DataFrame): [description]
            train_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.

        Returns:
            DataFrame of metrics, model name, target column and identifiers columns.
        """
        item_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_config.get_label()
        agg_metrics[METRICS_DATASET.MODEL_COLUMN] = self.model_config.get_label()

        identifiers_columns = (
            list(train_list_dataset.list_data[0][TIMESERIES_KEYS.IDENTIFIERS].keys())
            if TIMESERIES_KEYS.IDENTIFIERS in train_list_dataset.list_data[0]
            else []
        )
        identifiers_values = {identifiers_column: [] for identifiers_column in identifiers_columns}
        target_columns = []
        for univariate_timeseries in train_list_dataset.list_data:
            target_columns += [univariate_timeseries[TIMESERIES_KEYS.TARGET_NAME]]
            for identifiers_column in identifiers_columns:
                identifiers_values[identifiers_column] += [
                    univariate_timeseries[TIMESERIES_KEYS.IDENTIFIERS][identifiers_column]
                ]

        item_metrics[METRICS_DATASET.TARGET_COLUMN] = target_columns
        agg_metrics[METRICS_DATASET.TARGET_COLUMN] = METRICS_DATASET.AGGREGATED_ROW
        agg_metrics[METRICS_DATASET.TRAINING_TIME] = self.evaluation_time + self.retraining_time

        for identifiers_column in identifiers_columns:
            item_metrics[identifiers_column] = identifiers_values[identifiers_column]
            agg_metrics[
                identifiers_column
            ] = METRICS_DATASET.AGGREGATED_ROW  # or keep empty but will cast integer to float

        metrics = item_metrics.append(agg_metrics, ignore_index=True)

        metrics = metrics[
            [METRICS_DATASET.TARGET_COLUMN]
            + identifiers_columns
            + [METRICS_DATASET.MODEL_COLUMN]
            + list(EVALUATION_METRICS_DESCRIPTIONS.keys())
            + [METRICS_DATASET.TRAINING_TIME]
        ]
        metrics[METRICS_DATASET.MODEL_PARAMETERS] = self._get_model_parameters_json(train_list_dataset)

        return metrics, identifiers_columns

    def _train_estimator(self, train_list_dataset):
        """Train a gluonTS estimator to get a predictor for models that can be trained
        or directly get the existing gluonTS predictor (e.g. for models that don't need training like trivial.identity)

        Args:
            train_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.

        Returns:
            gluonts.model.predictor.Predictor
        """
        kwargs = {"freq": self.frequency, "prediction_length": self.prediction_length}
        if self.estimator is None:
            if self.model_config.needs_num_samples():
                kwargs.update({"num_samples": 100})
            if self.use_seasonality and self.season_length:
                kwargs.update({"season_length": self.season_length})
            predictor = self.model_config.predictor(**kwargs)
        else:
            try:
                predictor = self.estimator.train(train_list_dataset)
            except Exception as err:
                raise ModelTrainingError(
                    f"GluonTS '{self.model_name}' model crashed during training. Full error: {err}"
                )
        return predictor

    def _get_model_parameters_json(self, train_list_dataset):
        """Returns a JSON string containing model parameters and results"""
        timeseries_number = len(train_list_dataset.list_data)
        timeseries_total_length = sum([len(ts[TIMESERIES_KEYS.TARGET]) for ts in train_list_dataset.list_data])
        model_params = {
            "model_name": self.model_config.get_label(),
            "frequency": self.frequency,
            "prediction_length": self.prediction_length,
            "use_external_features": self.use_external_features,
            "timeseries_number": timeseries_number,
            "timeseries_average_length": round(timeseries_total_length / timeseries_number),
            "model_parameters": self.model_parameters,
        }
        if self.trainer is not None:
            model_params["epoch"] = self.epoch
            model_params["batch_size"] = self.batch_size
            model_params["num_batches_per_epoch"] = self.num_batches_per_epoch
        if self.use_seasonality and self.season_length is not None:
            model_params["season_length"] = self.season_length
        if self.mxnet_context:
            model_params["mxnet.context"] = str(self.mxnet_context)
        return json.dumps(model_params)

    def _compute_median_forecasts_timeseries(self, forecasts_list, train_list_dataset):
        """Compute median forecasts timeseries for each Forecast of forecasts_list.

        Args:
            forecasts_list (list): List of gluonts.model.forecast.Forecast (objects storing the predicted distributions as samples).
            train_list_dataset (gluonts.dataset.common.ListDataset): ListDataset created with the DkuGluonDataset class.

        Returns:
            Dictionary of list of forecasts timeseries (value) by identifiers (key). Key is None if no identifiers.
        """
        median_forecasts_timeseries = {}
        for i, sample_forecasts in enumerate(forecasts_list):
            series = quantile_forecasts_series(sample_forecasts, 0.5, self.custom_frequency).rename(
                f"{self.model_name}_{train_list_dataset.list_data[i][TIMESERIES_KEYS.TARGET_NAME]}"
            )
            if TIMESERIES_KEYS.IDENTIFIERS in train_list_dataset.list_data[i]:
                timeseries_identifier_key = tuple(
                    sorted(train_list_dataset.list_data[i][TIMESERIES_KEYS.IDENTIFIERS].items())
                )
            else:
                timeseries_identifier_key = None

            if timeseries_identifier_key in median_forecasts_timeseries:
                median_forecasts_timeseries[timeseries_identifier_key] += [series]
            else:
                median_forecasts_timeseries[timeseries_identifier_key] = [series]
        return median_forecasts_timeseries
