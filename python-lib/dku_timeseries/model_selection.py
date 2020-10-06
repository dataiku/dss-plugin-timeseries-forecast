import re
from plugin_io_utils import read_csv_from_folder, read_pickle_from_folder


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
        model = read_pickle_from_folder(model_path, self.folder)  # TODO implement load_model
        return model

    def get_training_dataframe(self):
        training_dataset_path = "{}/training_time_series.csv".format(self.session)
        df = read_csv_from_folder(training_dataset_path, self.folder)
        return df

    def _get_last_session(self):
        session_timestamps = []
        for child in self.folder.get_path_details(path='/')['children']:
            if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', child['name']):
                session_timestamps += [child['name']]
        last_session = max(session_timestamps, key=lambda timestamp: int(timestamp.replace('-', '')))
        return last_session

    def _get_best_model(self):
        # TODO make it work
        df = read_csv_from_folder("{}/model_results.csv".format(self.session), self.folder)  # TODO implement load_csv
        model_type = df.loc[df[self.performance_metric].idxmin()]['model']  # or idxmin() if minimize metric
        return model_type
