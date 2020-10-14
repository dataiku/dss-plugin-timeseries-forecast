import re
from plugin_io_utils import read_from_folder
from plugin_config_loading import PluginParamValidationError


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

        model_path = "{}/{}/model.pk".format(self.session, self.model_type)
        model = read_from_folder(self.folder, model_path, 'pickle')  # TODO implement load_model
        return model

    def get_targets_train_dataframe(self):
        targets_train_dataset_path = "{}/targets_train_dataset.csv.gz".format(self.session)
        targets_train_df = read_from_folder(self.folder, targets_train_dataset_path, 'csv.gz')
        return targets_train_df

    def get_external_features_dataframe(self):
        external_features_train_dataset_path = "{}/external_features_train_dataset.csv.gz".format(self.session)
        path_details = self.folder.get_path_details(external_features_train_dataset_path)
        trained_with_external_features = path_details['exists'] and not path_details['directory']
        if self.external_features_future_dataset:
            if trained_with_external_features:
                external_features_train_df = read_from_folder(self.folder, external_features_train_dataset_path, 'csv.gz')
                external_features_future_df = self.external_features_future_dataset.get_dataframe()
                if set(external_features_train_df.columns) != set(external_features_future_df.columns):
                    raise PluginParamValidationError("External features dataset must exactly contain the following columns: {}".format(
                        external_features_train_df.columns))
                # TODO ? check that the time column is just 1 freq after
                return external_features_train_df.append(external_features_future_df)
            else:
                # maybe no exception as it won't cause any errors (instead just return None ?)
                raise PluginParamValidationError("An external features dataset was provided forprediction but no external features were used in training.")
        else:
            if trained_with_external_features:
                raise PluginParamValidationError("No external features dataset was provided for prediction but some external features were used in training.")
            return None

    def _get_last_session(self):
        session_timestamps = []
        for child in self.folder.get_path_details(path='/')['children']:
            if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', child['name']):
                session_timestamps += [child['name']]
        last_session = max(session_timestamps, key=lambda timestamp: int(timestamp.replace('-', '')))
        return last_session

    def _get_best_model(self):
        df = read_from_folder(self.folder, "{}/metrics.csv".format(self.session), 'csv')
        if (df['target_col'] == 'AGGREGATED').any():
            df = df[df['target_col'] == 'AGGREGATED']
        assert df['model'].nunique() == len(df.index), "More than one row per model"
        model_type = df.loc[df[self.performance_metric].idxmin()]['model']  # or idxmax() if maximize metric
        return model_type
