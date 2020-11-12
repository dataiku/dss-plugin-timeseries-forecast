import pandas as pd
from gluonts_forecasts.model import Model
from constants import METRICS_DATASET
from gluonts_forecasts.gluon_dataset import GluonDataset


class TrainingSession:
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
        gpu=None,
        context_length=None,
    ):
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
        self.use_external_features = len(external_features_columns_names) > 0
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.version_name = None
        if self.make_forecasts:
            self.forecasts_df = pd.DataFrame()
            self.evaluation_forecasts_df = None
        self.train_list_dataset = None
        self.test_list_dataset = None
        self.metrics_df = None
        self.batch_size = batch_size
        self.gpu = gpu
        self.context_length = context_length

    def init(self, version_name, partition_root=None):
        if partition_root is None:
            self.version_name = version_name
        else:
            self.version_name = "{}/{}".format(partition_root, version_name)
        self.models = []
        for model_name in self.models_parameters:
            model_parameters = self.models_parameters.get(model_name)
            self.models.append(
                Model(
                    model_name,
                    model_parameters=model_parameters,
                    frequency=self.frequency,
                    prediction_length=self.prediction_length,
                    epoch=self.epoch,
                    use_external_features=self.use_external_features,
                    batch_size=self.batch_size,
                    gpu=self.gpu,
                    context_length=self.context_length,
                )
            )
        self.training_df[self.time_column_name] = pd.to_datetime(self.training_df[self.time_column_name]).dt.tz_localize(tz=None)

    def train(self):
        for model in self.models:
            model.train(self.test_list_dataset)

    def evaluate(self, evaluation_strategy):
        gluon_dataset = GluonDataset(
            dataframe=self.training_df,
            time_column_name=self.time_column_name,
            frequency=self.frequency,
            target_columns_names=self.target_columns_names,
            timeseries_identifiers_names=self.timeseries_identifiers_names,
            external_features_columns_names=self.external_features_columns_names,
        )

        if evaluation_strategy == "split":
            self.train_list_dataset = gluon_dataset.create_list_dataset(cut_length=self.prediction_length)
            self.test_list_dataset = gluon_dataset.create_list_dataset()
        else:
            raise Exception("{} evaluation strategy not implemented".format(evaluation_strategy))
        self.metrics_df = self._compute_all_evaluation_metrics()

    def _compute_all_evaluation_metrics(self):
        metrics_df = pd.DataFrame()
        for model in self.models:
            if self.make_forecasts:
                (
                    agg_metrics,
                    item_metrics,
                    forecasts_df,
                    identifiers_columns,
                ) = model.evaluate(self.train_list_dataset, self.test_list_dataset, make_forecasts=True)
                forecasts_df = forecasts_df.rename(columns={"index": self.time_column_name})
                if self.forecasts_df.empty:
                    self.forecasts_df = forecasts_df
                else:
                    self.forecasts_df = self.forecasts_df.merge(forecasts_df, on=[self.time_column_name] + identifiers_columns)
            else:
                agg_metrics, item_metrics = model.evaluate(self.train_list_dataset, self.test_list_dataset)
            metrics_df = metrics_df.append(item_metrics)
        metrics_df["session"] = self.version_name
        orderd_metrics_df = self._reorder_metrics_df(metrics_df)

        if self.make_forecasts:
            self.evaluation_forecasts_df = self.training_df.merge(
                self.forecasts_df,
                on=[self.time_column_name] + identifiers_columns,
                how="left",
            )
            self.evaluation_forecasts_df["session"] = self.version_name

        return orderd_metrics_df

    def _reorder_metrics_df(self, metrics_df):
        """ sort rows by target column and put aggregated rows on top """
        metrics_df = metrics_df.sort_values(by=[METRICS_DATASET.TARGET_COLUMN], ascending=True)
        orderd_metrics_df = pd.concat(
            [
                metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN] == METRICS_DATASET.AGGREGATED_ROW],
                metrics_df[metrics_df[METRICS_DATASET.TARGET_COLUMN] != METRICS_DATASET.AGGREGATED_ROW],
            ],
            axis=0,
        ).reset_index(drop=True)
        return orderd_metrics_df

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
