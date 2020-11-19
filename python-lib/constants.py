class METRICS_DATASET:
    """ Class of constants with labels used in the evaluation metrics dataframe of the train recipe """

    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"
    MODEL_PARAMETERS = "model_params"
    SESSION = "session"


class TIMESERIES_KEYS:
    """ Class of constants with labels for the keys used in the timeseries of the GluonDataset class """

    START = "start"
    TARGET = "target"
    TARGET_NAME = "target_name"
    TIME_COLUMN_NAME = "time_column_name"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"
    FEAT_DYNAMIC_REAL_COLUMNS_NAMES = "feat_dynamic_real_columns_names"
    IDENTIFIERS = "identifiers"


EVALUATION_METRICS_DESCRIPTIONS = {
    "MSE": "Mean Squared Error",
    "MASE": "Mean Absolute Scaled Error",
    "MAPE": "Mean Absolute Percentage Error",
    "sMAPE": "Symmetric Mean Absolute Percentage Error",
    "MSIS": "Mean Scaled Interval Score",
    "RMSE": "Root Mean Square Error",
    "ND": "Normalized Deviation",
    "mean_wQuantileLoss": "Mean Weight Quantile Loss",
}

METRICS_COLUMNS_DESCRIPTIONS = {
    METRICS_DATASET.MODEL_PARAMETERS: "Parameters used for training",
    METRICS_DATASET.SESSION: "Timestamp of training session",
    METRICS_DATASET.TARGET_COLUMN: "'AGGREGATED' to aggregate metrics of different targets within a model",
}

TIMESTAMP_REGEX_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z"
