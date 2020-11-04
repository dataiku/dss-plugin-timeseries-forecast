import re
from plugin_io_utils import read_from_folder, METRICS_DATASET
import pandas as pd
from dku_timeseries.model_descriptor import MODEL_DESCRIPTORS, CAN_USE_EXTERNAL_FEATURES

# TODO move file outside of dku_timeseries as it interacts with dataiku folder
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

        # TODO raise explicit error if selected model is not in selected session
        model_path = "{}/{}/model.pk.gz".format(self.session, self.model_type)
        model = read_from_folder(self.folder, model_path, 'pickle.gz')
        return model

    def get_gluon_train_dataset(self):
        gluon_train_dataset_path = "{}/gluon_train_dataset.pickle.gz".format(self.session)
        gluon_train_dataset = read_from_folder(self.folder, gluon_train_dataset_path, 'pickle.gz')
        return gluon_train_dataset

    # def get_targets_train_dataframe(self):
    #     targets_train_dataset_path = "{}/targets_train_dataset.csv.gz".format(self.session)
    #     targets_train_df = read_from_folder(self.folder, targets_train_dataset_path, 'csv.gz')
    #     targets_train_df.iloc[:, 0] = pd.to_datetime(targets_train_df.iloc[:, 0]).dt.tz_localize(tz=None)
    #     return targets_train_df

    # def get_external_features_dataframe(self):
    #     external_features_train_dataset_path = "{}/external_features_train_dataset.csv.gz".format(self.session)
    #     path_details = self.folder.get_path_details(external_features_train_dataset_path)
    #     trained_with_external_features = path_details['exists'] and not path_details['directory']
    #     if self.external_features_future_dataset:
    #         if trained_with_external_features:
    #             # the external features won't be used if the model doesn't support feat_dynamic_real
    #             # we could either raise an Error to warn the user or do as usual and output the external
    #             # features in the forecasts dataset even though they are not used
    #             external_features_train_df = read_from_folder(self.folder, external_features_train_dataset_path, 'csv.gz')
    #             external_features_future_df = self.external_features_future_dataset.get_dataframe()
    #             if not set(external_features_train_df.columns) <= set(external_features_future_df.columns):
    #                 raise ValueError("External features dataset must contain the following columns: {}".format(
    #                     external_features_train_df.columns))
    #             external_features_future_df = external_features_future_df[external_features_train_df.columns]
    #             # convert to same datetime format before appending past and future exteranl features dataframes
    #             external_features_train_df.iloc[:, 0] = pd.to_datetime(external_features_train_df.iloc[:, 0]).dt.tz_localize(tz=None)
    #             external_features_future_df.iloc[:, 0] = pd.to_datetime(external_features_future_df.iloc[:, 0]).dt.tz_localize(tz=None)

    #             # TODO ? check that the time column is just 1 freq after (using self.model.freq)
    #             return external_features_train_df.append(external_features_future_df).reset_index(drop=True)
    #         else:
    #             # it would not cause any errors to return None but the ouput would not contain the external features,
    #             # that could be confusing for the user => raise Error for transparency
    #             raise ValueError("""
    #                             An external features dataset was provided for forecasting but no external features
    #                             were used for training in the selected session. Please remove the external features
    #                             Dataset or select a model that was trained with external features.
    #                         """)
    #     else:
    #         if trained_with_external_features and MODEL_DESCRIPTORS[self.model_type].get(CAN_USE_EXTERNAL_FEATURES):
    #             raise ValueError("""
    #                         No external features dataset was provided for forecasting but some external features
    #                         were used in training to train the selected model. Please provide an external features
    #                         Dataset or select a model that doesn't use external features.
    #                     """)
    #         return None

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
