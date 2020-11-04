import re
from plugin_io_utils import read_from_folder, METRICS_DATASET
import pandas as pd
from dku_timeseries.model_descriptor import MODEL_DESCRIPTORS, CAN_USE_EXTERNAL_FEATURES

# TODO ? move file outside of dku_timeseries as it interacts with dataiku folder
class ModelSelection():

    def __init__(self, folder, external_features_future_dataset=None):
        self.folder = folder
        self.external_features_future_dataset = external_features_future_dataset

    def manual_params(self, session, model_type):
        self.manual_selection = True
        self.session = session
        self.model_type = model_type

    def auto_params(self, performance_metric):
        self.manual_selection = False
        self.performance_metric = performance_metric

    def get_model(self):
        if not self.manual_selection:
            self.session = self._get_last_session()
            self.model_type = self._get_best_model()

        model_path = "{}/{}/model.pk.gz".format(self.session, self.model_type)
        model = read_from_folder(self.folder, model_path, 'pickle.gz')
        return model

    def get_gluon_train_dataset(self):
        gluon_train_dataset_path = "{}/gluon_train_dataset.pickle.gz".format(self.session)
        gluon_train_dataset = read_from_folder(self.folder, gluon_train_dataset_path, 'pickle.gz')
        return gluon_train_dataset

    def _get_last_session(self):
        session_timestamps = []
        for child in self.folder.get_path_details(path='/')['children']:
            if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z', child['name']):
                session_timestamps += [child['name']]
        last_session = max(session_timestamps, key=lambda timestamp: timestamp)
        return last_session

    def _get_best_model(self):
        df = read_from_folder(self.folder, "{}/metrics.csv".format(self.session), 'csv')
        if (df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW).any():
            df = df[df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW]
        assert df[METRICS_DATASET.MODEL_COLUMN].nunique() == len(df.index), "More than one row per model"
        model_type = df.loc[df[self.performance_metric].idxmin()][METRICS_DATASET.MODEL_COLUMN]  # or idxmax() if maximize metric
        return model_type
