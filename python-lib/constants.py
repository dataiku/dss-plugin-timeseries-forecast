from gluonts.dataset.field_names import FieldName


class METRICS_DATASET:
    """ Class of constants with labels used in the evaluation metrics dataframe of the train recipe """

    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "__aggregated__"
    MODEL_PARAMETERS = "model_params"
    SESSION = "training_session"
    TRAINING_TIME = "training_time"


class TIMESERIES_KEYS:
    """ Class of constants with labels for the keys used in the timeseries of the GluonDataset class """

    START = FieldName.START
    TARGET = FieldName.TARGET
    TARGET_NAME = "target_name"
    TIME_COLUMN_NAME = "time_column_name"
    FEAT_DYNAMIC_REAL = FieldName.FEAT_DYNAMIC_REAL
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
    METRICS_DATASET.MODEL_COLUMN: "Model name",
    METRICS_DATASET.MODEL_PARAMETERS: "Parameters used for training",
    METRICS_DATASET.SESSION: "Timestamp of training session",
    METRICS_DATASET.TARGET_COLUMN: "Aggregated and per-time-series metrics",
    METRICS_DATASET.TRAINING_TIME: "Time to train model for evaluation (in seconds)",
}

# regex pattern to match the timestamps used for training sessions
TIMESTAMP_REGEX_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z"


FORECASTING_STYLE_PRESELECTED_MODELS = {
    "auto": ["naive", "deepar"],
    "auto_performance": ["naive", "deepar", "transformer"]
    }
