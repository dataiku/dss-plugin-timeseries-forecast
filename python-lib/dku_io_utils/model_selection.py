import re
import os
from dku_io_utils.utils import read_from_folder
from dku_constants import METRICS_DATASET, TIMESTAMP_REGEX_PATTERN, ObjectType
from gluonts_forecasts.model_config_registry import ModelConfigRegistry
from timeseries_preparation.preparation import TimeseriesPreparator
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


class ModelSelectionError(ValueError):
    """Custom exception raised when the predict recipe is unable to retrieve a model"""

    pass


class ModelSelection:
    """
    Class to retrieve from the input folder the saved trained model and the training gluon dataset

    Attributes:
        folder (dataiku.Folder): Input folder containing trained models and training data
        manual_selection (bool): True if session and model are manually selected by user
        performance_metric (str): Name of evaluation metric used to select best model
        session_name (str): Timestamp of training session
        model_label (str): Label of trained model
        partition_root (str, optional): Partition root path (empty if no partitioning)
        predictors (dict): Dict of GluonTS Predictor objects (value) by model label (key)
    """

    def __init__(self, folder, manual_selection, performance_metric, session_name, model_label, partition_root=None):
        self.folder = folder
        self.partition_root = "" if not partition_root else partition_root

        if session_name == "latest_session" or not manual_selection:
            self.session_name = self._get_last_session()
        else:
            self.session_name = session_name

        self.session_path = os.path.join(self.partition_root, self.session_name)

        if manual_selection:
            self.model_label = model_label
        else:
            self.model_label = self._get_best_model(performance_metric)

        self.predictors = {}
        if self.model_label == "all_models":
            models_labels = self.find_all_models_labels_from_folder(self.folder, self.session_path, error_if_empty=True)
            for model_label in models_labels:
                self.predictors[model_label] = self._get_model_predictor(model_label)
        else:
            self.predictors[self.model_label] = self._get_model_predictor(self.model_label)

        first_predictor = self.get_first_predictor()
        self._prediction_length = first_predictor.prediction_length
        self._frequency = first_predictor.freq

    def get_session_name(self):
        return self.session_name

    def get_model_predictors(self):
        return self.predictors

    def _get_model_predictor(self, model_label):
        """Retrieve the GluonTS Predictor object obtained during training and saved into the model folder"""
        model_path = os.path.join(self.session_path, model_label, "model.pk.gz")
        try:
            model = read_from_folder(self.folder, model_path, ObjectType.PICKLE_GZ)
        except ValueError as e:
            raise ModelSelectionError(
                f"Unable to retrieve model '{model_label}' from session '{self.session_name}'. "
                + f"Please make sure that it exists in the Trained model folder. Full error: {e}"
            )
        return model

    def get_gluon_train_list_dataset(self):
        """Retrieve the DkuGluonDataset object with training data that was saved in the model folder during training"""
        gluon_train_dataset_path = f"{self.session_path}/gluon_train_dataset.pk.gz"
        gluon_train_dataset = read_from_folder(self.folder, gluon_train_dataset_path, ObjectType.PICKLE_GZ)
        return gluon_train_dataset

    def get_timeseries_preparator(self):
        timeseries_preparator_path = os.path.join(self.session_path, "timeseries_preparator.json")
        try:
            timeseries_preparator_serialized = read_from_folder(
                self.folder, timeseries_preparator_path, ObjectType.JSON
            )
            timeseries_preparator = TimeseriesPreparator.deserialize(timeseries_preparator_serialized)

        except (ValueError, TypeError) as e:
            raise ModelSelectionError(
                f"""Unable to load training timeseries metadata at path '{timeseries_preparator_path}' from session '{self.session_name}'.
                Please make sure that it exists in the Trained model folder and was generated by the training recipe.
                Full error: {e}
                """
            )
        return timeseries_preparator

    def get_prediction_length(self):
        return self._prediction_length

    def get_frequency(self):
        return self._frequency

    def _get_last_session(self):
        """Retrieve the last training session using name of subfolders and append the partition root path.

        Returns:
            Timestamp of the last training session.
        """
        session_timestamps = []
        for child in self.folder.get_path_details(path=self.partition_root)["children"]:
            if re.match(TIMESTAMP_REGEX_PATTERN, child["name"]):
                session_timestamps += [child["name"]]
        if len(session_timestamps) == 0:
            raise ModelSelectionError(f"Model not found in {self.partition_root}")
        last_session = max(session_timestamps, key=lambda timestamp: timestamp)
        return last_session

    def _get_best_model(self, performance_metric):
        """Find the best model according to performance_metric based on the aggregated metric rows
        (rows where both the target and the rolling window index are aggregated)

        Returns:
            Label of the best model.
        """
        available_models_labels = ModelConfigRegistry().list_available_models_labels()
        df = read_from_folder(self.folder, f"{self.session_path}/metrics.csv", ObjectType.CSV)
        try:
            if (df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW).any():
                df = df[df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW]

            assert df[METRICS_DATASET.MODEL_COLUMN].nunique() == len(df.index), "More than one row per model"
            model_label = df.loc[df[performance_metric].idxmin()][
                METRICS_DATASET.MODEL_COLUMN
            ]  # or idxmax() if maximize metric
            assert model_label in available_models_labels, "Best model retrieved is not an available models"
        except Exception as e:
            raise ModelSelectionError(
                f"Unable to find the best model of session '{self.session_name}' with the performance metric '{performance_metric}'. Full error: {e}"
            )
        return model_label

    @staticmethod
    def find_all_models_labels_from_folder(folder, session_path=None, error_if_empty=False):
        model_labels = []
        all_paths = folder.list_paths_in_partition()
        for model_label in ModelConfigRegistry().list_available_models_labels():
            for path in all_paths:
                path = path.strip("/")
                if session_path and not path.startswith(session_path):
                    continue
                if path.endswith(f"{model_label}/model.pk.gz"):
                    model_labels += [model_label]
                    break
        if len(model_labels) == 0:
            error_message = "No trained model found in in the Trained model folder"
            if session_path:
                error_message += f" at path: {session_path}"
            logger.warning(error_message)
            if error_if_empty:
                raise ModelSelectionError(error_message)
        return model_labels

    def get_first_predictor(self):
        """Retrieve the first predictor of the predictors dictionary"""
        if len(self.predictors) == 0:
            raise ModelSelectionError(f"No model found in session '{self.session_name}'")
        return next(iter(self.predictors.values()))
