from dku_constants import TIMESERIES_KEYS
from safe_logger import SafeLogger


logger = SafeLogger("Forecast plugin")


def external_features_future_dataset_schema_check(train_data_sample, external_features_future_dataset):
    """Check that the schema of external_features_future_dataset contains the following column names:
        - train_data_sample["time_column_name"]
        - train_data_sample["feat_dynamic_real_columns_names"]
        - train_data_sample["identifiers"].keys()

    Args:
        train_data_sample (dict): univariate timeseries dictionary
        external_features_future_dataset (dataiku.Dataset)

    Raises:
        ValueError: If the external_features_future_dataset doesn't have the right schema.
    """
    external_features_future_columns = [column["name"] for column in external_features_future_dataset.read_schema()]
    expected_columns = [train_data_sample[TIMESERIES_KEYS.TIME_COLUMN_NAME]] + train_data_sample[
        TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES
    ]
    if TIMESERIES_KEYS.IDENTIFIERS in train_data_sample:
        expected_columns += list(train_data_sample[TIMESERIES_KEYS.IDENTIFIERS].keys())
    if not set(expected_columns).issubset(set(external_features_future_columns)):
        raise ValueError(
            f"Dataset of future values of external features must contain the following columns: {expected_columns}"
        )


def external_features_check(gluon_train_dataset, external_features_future_dataset):
    """Check that an external features dataset with the right schema was provided if and only if external features were used during training

    Args:
        gluon_train_dataset (ListDataset): Dataset used for training
        external_features_future_dataset (dataiku.Dataset): Dataset of future values of external features

    Raises:
        ValueError:
            If gluon_train_dataset contains external features and no external_features_future_dataset was provided.
            If gluon_train_dataset doesn't contain external features but a external_features_future_dataset was provided.

    Returns:
        True if external features are needed for prediction, else False
    """
    train_data_sample = gluon_train_dataset.list_data[0]
    trained_with_external_features = TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES in train_data_sample
    if trained_with_external_features and external_features_future_dataset:
        external_features_future_dataset_schema_check(train_data_sample, external_features_future_dataset)
        return True
    elif trained_with_external_features and not external_features_future_dataset:
        raise ValueError(
            "Please provide a dataset of future values of external features in the 'Input / Output' tab of the recipe"
        )
    elif not trained_with_external_features and external_features_future_dataset:
        logger.warning(
            """A dataset of future values of external features was provided, but no external features were used when training the selected model."""
        )
    return False
