import pandas as pd
from dku_timeseries.single_model import SingleModel
from gluonts.dataset.common import ListDataset
from plugin_io_utils import write_to_folder, METRICS_DATASET


class GlobalModels():
    def __init__(self, target_columns_names, time_column_name, frequency, epoch, models_parameters, prediction_length,
                 training_df, make_forecasts, external_features_columns_names=None, timeseries_identifiers_names=None,
                 batch_size=None, gpu=None):
        self.models_parameters = models_parameters
        self.model_names = []
        self.models = None
        self.glutonts_dataset = None
        self.training_df = training_df
        self.prediction_length = prediction_length
        self.target_columns_names = target_columns_names
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.epoch = epoch
        self.make_forecasts = make_forecasts
        self.external_features_columns_names = external_features_columns_names
        self.use_external_features = (len(external_features_columns_names) > 0)
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.version_name = None
        if self.make_forecasts:
            self.forecasts_df = pd.DataFrame()
            self.evaluation_forecasts_df = None
        self.train_ds = None
        self.test_ds = None
        self.metrics_df = None
        self.batch_size = batch_size
        self.gpu = gpu

    def init_all_models(self, version_name, partition_root=None):
        if partition_root is None:
            self.version_name = version_name
        else:
            self.version_name = "{}/{}".format(partition_root, version_name)
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                SingleModel(
                    model_name,
                    model_parameters=model_parameters,
                    frequency=self.frequency,
                    prediction_length=self.prediction_length,
                    epoch=self.epoch,
                    use_external_features=self.use_external_features,
                    batch_size=self.batch_size,
                    gpu=self.gpu
                )
            )
        # already done in assert_continuous_time_column
        # self.training_df[self.time_column_name] = pd.to_datetime(self.training_df[self.time_column_name]).dt.tz_localize(tz=None)

    def fit_all(self):
        for model in self.models:
            model.fit(self.test_ds)

    def create_gluon_dataset(self, remove_length=None):
        length = -remove_length if remove_length else None
        multivariate_timeseries = []
        if self.timeseries_identifiers_names:
            # TODO check that all timeseries have same length
            for identifiers_values, identifiers_df in self.training_df.groupby(self.timeseries_identifiers_names):
                multivariate_timeseries += self._create_gluon_multivariate_timeseries(identifiers_df, length, identifiers_values=identifiers_values)
        else:
            multivariate_timeseries += self._create_gluon_multivariate_timeseries(self.training_df, length)
        # return multivariate_timeseries
        return ListDataset(multivariate_timeseries, freq=self.frequency)

    def _create_gluon_multivariate_timeseries(self, df, length, identifiers_values=None):
        multivariate_timeseries = []
        for target_column_name in self.target_columns_names:
            multivariate_timeseries.append(self._create_gluon_univariate_timeseries(df, target_column_name, length, identifiers_values))
        return multivariate_timeseries
    
    def _create_gluon_univariate_timeseries(self, df, target_column_name, length, identifiers_values=None):
        """ create dictionary for one timeseries and add extra features and identifiers if specified """
        univariate_timeseries = {
            'start': df[self.time_column_name].iloc[0],
            'target': df[target_column_name].iloc[:length].values,
            'target_name': target_column_name,
            'time_column_name': self.time_column_name
        }
        if self.external_features_columns_names:
            univariate_timeseries['feat_dynamic_real'] = df[self.external_features_columns_names].iloc[:length].values.T
            univariate_timeseries['feat_dynamic_real_columns_names'] = self.external_features_columns_names
        if identifiers_values:
            if len(self.timeseries_identifiers_names) > 1:
                identifiers_map = {self.timeseries_identifiers_names[i]: identifier_value for i, identifier_value in enumerate(identifiers_values)}
            else:
                identifiers_map = {self.timeseries_identifiers_names[0]: identifiers_values}
            univariate_timeseries['identifiers'] = identifiers_map
        return univariate_timeseries

    def evaluate_all(self, evaluation_strategy):
        if evaluation_strategy == "split":
            self.train_ds = self.create_gluon_dataset(remove_length=self.prediction_length)  # remove last prediction_length time steps
            self.test_ds = self.create_gluon_dataset()  # all time steps
        else:
            raise Exception("{} evaluation strategy not implemented".format(evaluation_strategy))
        self.metrics_df = self._compute_all_evaluation_metrics()

    def _compute_all_evaluation_metrics(self):
        metrics_df = pd.DataFrame()
        for model in self.models:
            if self.make_forecasts:
                agg_metrics, item_metrics, forecasts_df, identifiers_columns = model.evaluate(self.train_ds, self.test_ds, make_forecasts=True)
                forecasts_df = forecasts_df.rename(columns={'index': self.time_column_name})
                if self.forecasts_df.empty:
                    self.forecasts_df = forecasts_df
                else:
                    self.forecasts_df = self.forecasts_df.merge(forecasts_df, on=[self.time_column_name] + identifiers_columns)
            else:
                agg_metrics, item_metrics = model.evaluate(self.train_ds, self.test_ds)
            metrics_df = metrics_df.append(item_metrics)
        metrics_df['session'] = self.version_name
        orderd_metrics_df = self._reorder_metrics_df(metrics_df)
        
        if self.make_forecasts:
            self.evaluation_forecasts_df = self.training_df.merge(self.forecasts_df, on=[self.time_column_name] + identifiers_columns, how='left')
            self.evaluation_forecasts_df['session'] = self.version_name

        return orderd_metrics_df

    def _reorder_metrics_df(self, metrics_df):
        """ sort rows by target column and put aggregated rows on top """
        metrics_df = metrics_df.sort_values(by=[METRICS_DATASET.TARGET_COLUMN], ascending=True)
        orderd_metrics_df = pd.concat([
            metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN]==METRICS_DATASET.AGGREGATED_ROW],
            metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN]!=METRICS_DATASET.AGGREGATED_ROW]
        ], axis=0).reset_index(drop=True)
        return orderd_metrics_df

    def save_all(self, model_folder):
        # TODO ? move outside of the class as it interacts with dataiku.Folder objects
        metrics_path = "{}/metrics.csv".format(self.version_name)
        write_to_folder(self.metrics_df, model_folder, metrics_path, 'csv')

        gluon_train_dataset_path = "{}/gluon_train_dataset.pickle.gz".format(self.version_name)
        write_to_folder(self.test_ds, model_folder, gluon_train_dataset_path, 'pickle.gz')

        for model in self.models:
            model.save(model_folder=model_folder, version_name=self.version_name)

    def get_evaluation_forecasts_df(self):
        return self.evaluation_forecasts_df

    def get_metrics_df(self):
        return self.metrics_df

    def create_metrics_column_description(self):
        column_descriptions = {}
        for column in self.metrics_df.columns:
            column_descriptions[column] = "TO FILL"
        return column_descriptions

    def create_evaluation_forecasts_column_description(self):
        column_descriptions = {}
        for column in self.evaluation_forecasts_df.columns:
            column_descriptions[column] = "TO FILL"
        return column_descriptions
