import re
from plugin_io_utils import read_from_folder


class ModelSelection():

    def __init__(self, folder):
        self.folder = folder

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

    def get_training_dataframe(self):
        training_dataset_path = "{}/train_dataset.csv".format(self.session)
        # TODO read the compress csv
        print("training_dataset_path: ", training_dataset_path)
        df = read_from_folder(self.folder, training_dataset_path, 'csv')
        return df

    def _get_last_session(self):
        session_timestamps = []
        for child in self.folder.get_path_details(path='/')['children']:
            if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', child['name']):
                session_timestamps += [child['name']]
        last_session = max(session_timestamps, key=lambda timestamp: int(timestamp.replace('-', '')))
        return last_session

    def _get_best_model(self):
        # TODO read the compress csv
        df = read_from_folder(self.folder, "{}/metrics.csv".format(self.session), 'csv')
        if (df['target_col'] == 'AGGREGATED').any():
            df = df[df['target_col'] == 'AGGREGATED']
        assert df['model'].nunique() == len(df.index), "More than one row per model"
        model_type = df.loc[df[self.performance_metric].idxmin()]['model']  # or idxmax() if maximize metric
        return model_type
