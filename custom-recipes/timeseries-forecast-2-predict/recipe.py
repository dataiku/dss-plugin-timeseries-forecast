from dataiku.customrecipe import get_recipe_config
from dku_io_utils.utils import set_column_description
from dku_io_utils.checks_utils import external_features_check
from dku_io_utils.model_selection import ModelSelection
from dku_io_utils.config_handler import create_dku_config
from dku_io_utils.dku_file_manager import DkuFileManager
from dku_constants import RECIPE, MIN_TRAIN_TO_TEST_LENGTH_RATIO
from gluonts_forecasts.utils import add_future_external_features
from gluonts_forecasts.trained_model import TrainedModel, predict_multiple_models
from gluonts_forecasts.gluon_dataset import DkuGluonDataset
from safe_logger import SafeLogger


def create_dku_file_manager():
    file_manager = DkuFileManager()
    file_manager.add_input_folder("model_folder")
    file_manager.add_input_dataset("historical_dataset", required=False)
    file_manager.add_input_dataset("external_features_future_dataset", required=False)
    file_manager.add_output_dataset("output_dataset")
    return file_manager


def run():
    logger = SafeLogger("Forecast plugin")
    logger.info("Forecasting future values...")

    recipe_config = get_recipe_config()
    file_manager = create_dku_file_manager()
    dku_config = create_dku_config(RECIPE.PREDICT, recipe_config, file_manager)

    model_selection = ModelSelection(
        folder=file_manager.model_folder,
        manual_selection=dku_config.manual_selection,
        performance_metric=dku_config.performance_metric,
        session_name=dku_config.selected_session,
        model_label=dku_config.selected_model_label,
        partition_root=dku_config.partition_root,
    )

    if file_manager.historical_dataset:
        timeseries_preparator = model_selection.get_timeseries_preparator()

        timeseries_preparator.check_schema_from_dataset(file_manager.historical_dataset.read_schema())

        historical_dataset_dataframe = file_manager.historical_dataset.get_dataframe()

        prepared_dataframe = timeseries_preparator.prepare_timeseries_dataframe(historical_dataset_dataframe)

        gluon_dataset = DkuGluonDataset(
            time_column_name=timeseries_preparator.time_column_name,
            frequency=timeseries_preparator.frequency,
            target_columns_names=timeseries_preparator.target_columns_names,
            timeseries_identifiers_names=timeseries_preparator.timeseries_identifiers_names,
            external_features_columns_names=timeseries_preparator.external_features_columns_names,
            min_length=MIN_TRAIN_TO_TEST_LENGTH_RATIO
            * (timeseries_preparator.prediction_length if timeseries_preparator.prediction_length else 1),
        )

        gluon_train_list_dataset = gluon_dataset.create_list_datasets(prepared_dataframe)[0]

    else:
        gluon_train_list_dataset = model_selection.get_gluon_train_list_dataset()

    has_external_features = external_features_check(
        gluon_train_list_dataset, file_manager.external_features_future_dataset
    )

    if has_external_features:
        external_features_future_df = file_manager.external_features_future_dataset.get_dataframe()
        gluon_train_list_dataset = add_future_external_features(
            gluon_train_list_dataset,
            external_features_future_df,
            model_selection.get_prediction_length(),
            model_selection.get_frequency(),
        )

    trained_model = TrainedModel(
        gluon_dataset=gluon_train_list_dataset,
        prediction_length=model_selection.get_prediction_length(),
        frequency=model_selection.get_frequency(),
        quantiles=dku_config.quantiles,
        include_history=dku_config.include_history,
        history_length_limit=dku_config.history_length_limit,
    )

    forecasts_df = predict_multiple_models(trained_model, model_selection.get_model_predictors())

    forecasts_df = trained_model.get_forecasts_df_for_display(forecasts_df, model_selection.get_session_name())

    file_manager.output_dataset.write_with_schema(forecasts_df)

    column_descriptions = trained_model.create_forecasts_column_description(forecasts_df)
    set_column_description(file_manager.output_dataset, column_descriptions)


run()
