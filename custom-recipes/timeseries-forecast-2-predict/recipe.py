from dataiku.customrecipe import get_recipe_config
from dku_io_utils.utils import set_column_description
from dku_io_utils.checks_utils import external_features_check
from dku_io_utils.model_selection import ModelSelection
from dku_io_utils.config_handler import create_dku_config
from dku_io_utils.dku_file_manager import DkuFileManager
from dku_constants import RECIPE
from gluonts_forecasts.utils import add_future_external_features
from gluonts_forecasts.trained_model import TrainedModel
from gluonts_forecasts.gluon_dataset import GluonDataset
from safe_logger import SafeLogger
from time import perf_counter


def create_dku_file_manager():
    file_manager = DkuFileManager()
    file_manager.add_input_folder("model_folder")
    file_manager.add_input_dataset("historical_dataset", required=False)
    file_manager.add_input_dataset("external_features_future_dataset", required=False)
    file_manager.add_output_dataset("output_dataset")
    return file_manager


def run():
    logger = SafeLogger("Forecast plugin")
    start = perf_counter()
    logger.info("Forecasting future values...")

    recipe_config = get_recipe_config()
    file_manager = create_dku_file_manager()
    dku_config = create_dku_config(RECIPE.PREDICT, recipe_config, file_manager)

    model_selection = ModelSelection(
        folder=file_manager.model_folder,
        partition_root=dku_config.partition_root,
    )

    if dku_config.manual_selection:
        model_selection.set_manual_selection_parameters(
            session_name=dku_config.selected_session, model_label=dku_config.selected_model_label
        )
    else:
        model_selection.set_auto_selection_parameters(performance_metric=dku_config.performance_metric)

    predictor = model_selection.get_model_predictor()

    if file_manager.historical_dataset:
        # note: because dataframe is sorted in preparation according to identifiers and time column, order is maintained if same timeseries identifiers identifiers

        timeseries_preparator = model_selection.get_timeseries_preparator()

        timeseries_preparator.check_schema_from_dataset(file_manager.historical_dataset.read_schema())

        historical_dataset_dataframe = file_manager.historical_dataset.get_dataframe()

        prepared_dataframe = timeseries_preparator.prepare_timeseries_dataframe(historical_dataset_dataframe)

        gluon_dataset = GluonDataset(
            time_column_name=timeseries_preparator.time_column_name,
            frequency=timeseries_preparator.frequency,
            target_columns_names=timeseries_preparator.target_columns_names,
            timeseries_identifiers_names=timeseries_preparator.timeseries_identifiers_names,
            external_features_columns_names=timeseries_preparator.external_features_columns_names,
            min_length=2 * (timeseries_preparator.prediction_length if timeseries_preparator.prediction_length else 1),
        )

        gluon_train_list_dataset = gluon_dataset.create_list_datasets(prepared_dataframe)[0]

    else:
        gluon_train_list_dataset = model_selection.get_gluon_train_list_dataset()

    external_features = external_features_check(gluon_train_list_dataset, file_manager.external_features_future_dataset)

    if external_features:
        external_features_future_df = file_manager.external_features_future_dataset.get_dataframe()
        gluon_train_list_dataset = add_future_external_features(
            gluon_train_list_dataset, external_features_future_df, predictor.prediction_length, predictor.freq
        )

    trained_model = TrainedModel(
        predictor=predictor,
        gluon_dataset=gluon_train_list_dataset,
        prediction_length=dku_config.prediction_length,
        quantiles=dku_config.quantiles,
        include_history=dku_config.include_history,
        history_length_limit=dku_config.history_length_limit,
        model_name=model_selection.get_model_name(),
    )

    trained_model.predict()

    logger.info("Forecasting future values: Done in {:.2f} seconds".format(perf_counter() - start))

    forecasts_df = trained_model.get_forecasts_df(session=model_selection.get_session_name())
    file_manager.output_dataset.write_with_schema(forecasts_df)

    column_descriptions = trained_model.create_forecasts_column_description()
    set_column_description(file_manager.output_dataset, column_descriptions)


run()
