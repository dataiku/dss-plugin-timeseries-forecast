def external_features_future_dataset_schema_check(
    train_data_sample, external_features_future_dataset
):
    """
    check that schema of external_features_future_dataset contains exactly and only
    time_column_name | feat_dynamic_real_columns_names | identifiers.keys()
    """
    external_features_future_columns = [
        column["name"] for column in external_features_future_dataset.read_schema()
    ]
    expected_columns = [train_data_sample["time_column_name"]] + train_data_sample[
        "feat_dynamic_real_columns_names"
    ]
    if "identifiers" in train_data_sample:
        expected_columns += list(train_data_sample["identifiers"].keys())
    if set(external_features_future_columns) != set(expected_columns):
        raise ValueError(
            "The dataset of future values of external features must contains exactly the following columns: {}".format(
                expected_columns
            )
        )


def external_features_check(gluon_train_dataset, external_features_future_dataset):
    """
    check that an external features dataset has been provided if and only external features were used during training
    return True if external features are needed for prediction
    """
    train_data_sample = gluon_train_dataset.list_data[0]
    trained_with_external_features = bool("feat_dynamic_real_columns_names" in train_data_sample)
    if trained_with_external_features and external_features_future_dataset:
        external_features_future_dataset_schema_check(
            train_data_sample, external_features_future_dataset
        )
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
