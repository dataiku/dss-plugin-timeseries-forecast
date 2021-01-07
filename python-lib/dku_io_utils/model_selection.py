import re
import os
from dku_io_utils.utils import read_from_folder
from constants import METRICS_DATASET, TIMESTAMP_REGEX_PATTERN
from gluonts_forecasts.model_handler import list_available_models_labels


class ModelSelectionError(ValueError):
    """Custom exception raised when the predict recipe is unable to retrieve a model"""

    pass


class ModelSelection:
    """
    Class to retrieve from the input folder the saved trained model and the training gluon dataset

    Attributes:
        folder (dataiku.Folder): Input folder containing trained models and training data
        root (str): Partition root path (empty if no partitioning)
        manual_selection (bool): True if session and model are manually selected by user
        session_name (str): Timestamp of training session
        model_label (str): Label of trained model
        performance_metric (str): Name of evaluation metric used to select best model
    """

    def __init__(self, folder, partition=None):
        self.folder = folder
        self.root = "" if partition is None else partition
        self.manual_selection = None
        self.session_name = None
        self.session_path = None
        self.model_label = None
        self.performance_metric = None

    def set_manual_selection_parameters(self, session_name, model_label):
        """Set the session and model label if there were manually selected in the recipe form/

        Args:
            session_name (str): Timestamp of the selected training session.
            model_label (str): Label of the selected model.
        """
        self.manual_selection = True
        if session_name == "latest_session":
            self.session_name = self._get_last_session()
        else:
            self.session_name = session_name
        self.session_path = os.path.join(self.root, self.session_name)
        self.model_label = model_label

    def set_auto_selection_parameters(self, performance_metric):
        """Set the performance metric to use in order to retrieve the best model of the last session.

        Args:
            performance_metric (str): Name of evaluation metric used to select best model
        """
        self.manual_selection = False
        self.performance_metric = performance_metric

    def get_model_label(self):
        return self.model_label

    def get_session_name(self):
        return self.session_name

    def get_model_predictor(self):
        """ Retrieve the GluonTS Predictor object obtained during training and saved into the model folder """
        if not self.manual_selection:
            self.session_name = self._get_last_session()
            self.session_path = os.path.join(self.root, self.session_name)
            self.model_label = self._get_best_model()

        model_path = os.path.join(self.session_path, self.model_label, "model.pk.gz")
        try:
            model = read_from_folder(self.folder, model_path, "pickle.gz")
        except ValueError as e:
            raise ModelSelectionError(
                f"Unable to retrieve model '{self.model_label}' from session '{self.session_name}'. "
                + f"Please make sure that it exists in the Trained model folder. Full error: {e}"
            )
        return model

    def get_gluon_train_dataset(self):
        """ Retrieve the GluonDataset object with training data that was saved in the model folder during training """
        gluon_train_dataset_path = f"{self.session_path}/gluon_train_dataset.pk.gz"
        gluon_train_dataset = read_from_folder(self.folder, gluon_train_dataset_path, "pickle.gz")
        return gluon_train_dataset

    def _get_last_session(self):
        """Retrieve the last training session using name of subfolders and append the partition root path.

        Returns:
            Timestamp of the last training session.
        """
        session_timestamps = []
        for child in self.folder.get_path_details(path=self.root)["children"]:
            if re.match(TIMESTAMP_REGEX_PATTERN, child["name"]):
                session_timestamps += [child["name"]]
        if len(session_timestamps) == 0:
            raise ModelSelectionError("Model not found for {}".format(self.root))
        last_session = max(session_timestamps, key=lambda timestamp: timestamp)
        return last_session

    def _get_best_model(self):
        """Find the best model according to self.performance_metric based on the aggregated metric rows

        Returns:
            Label of the best model.
        """
        available_models_labels = list_available_models_labels()
        df = read_from_folder(self.folder, f"{self.session_path}/metrics.csv", "csv")
        try:
            if (df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW).any():
                df = df[df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW]
            assert df[METRICS_DATASET.MODEL_COLUMN].nunique() == len(df.index), "More than one row per model"
            model_label = df.loc[df[self.performance_metric].idxmin()][
                METRICS_DATASET.MODEL_COLUMN
            ]  # or idxmax() if maximize metric
            assert model_label in available_models_labels, "Best model retrieved is not an available models"
        except Exception as e:
            raise ModelSelectionError(
                f"Unable to find the best model of session '{self.session_name}' with the performance metric '{self.performance_metric}'. Full error: {e}"
            )
        return model_label
