from constants import TIMESERIES_KEYS


def external_features_future_dataset_schema_check(train_data_sample, external_features_future_dataset):
    """Check that the schema of external_features_future_dataset contains only the following column names:
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
    expected_columns = [train_data_sample[TIMESERIES_KEYS.TIME_COLUMN_NAME]] + train_data_sample[TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES]
    if TIMESERIES_KEYS.IDENTIFIERS in train_data_sample:
        expected_columns += list(train_data_sample[TIMESERIES_KEYS.IDENTIFIERS].keys())
    if set(external_features_future_columns) != set(expected_columns):
        raise ValueError("The dataset of future values of external features must contains exactly the following columns: {}".format(expected_columns))


def external_features_check(gluon_train_dataset, external_features_future_dataset):
    """Check that an external features dataset with the right schema was provided if and only external features were used during training

    Args:
        gluon_train_dataset ([type]): [description]
        external_features_future_dataset ([type]): [description]

    Raises:
        ValueError: If gluon_train_dataset contains external features and no external_features_future_dataset was provided.
        ValueError: If gluon_train_dataset doesn't contain external features but a external_features_future_dataset was provided.

    Returns:
        True if external features are needed for prediction, else False
    """
    train_data_sample = gluon_train_dataset.list_data[0]
    trained_with_external_features = TIMESERIES_KEYS.FEAT_DYNAMIC_REAL_COLUMNS_NAMES in train_data_sample
    if trained_with_external_features and external_features_future_dataset:
        external_features_future_dataset_schema_check(train_data_sample, external_features_future_dataset)
        return True
    elif trained_with_external_features and not external_features_future_dataset:
        raise ValueError("You must provide a dataset of future values of external features.")
    elif not trained_with_external_features and external_features_future_dataset:
        raise ValueError(
            """
            A dataset of future values of external features was provided, but no external features were used during training for the selected model.
            Remove this dataset from the recipe inputs or select a model that used external features during training.
        """
        )
    return False
