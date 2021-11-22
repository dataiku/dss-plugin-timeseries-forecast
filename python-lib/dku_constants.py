from gluonts.dataset.field_names import FieldName
from pandas.tseries.offsets import Week
from enum import Enum


class RECIPE(Enum):
    TRAIN_EVALUATE = "train_evaluate"
    PREDICT = "predict"


class ObjectType(Enum):
    CSV = "csv"
    CSV_GZ = "csv.gz"
    PICKLE = "pickle"
    PICKLE_GZ = "pickle.gz"
    JSON = "json"


class METRICS_DATASET:
    """Class of constants with labels used in the evaluation metrics dataframe of the train recipe"""

    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "__aggregated__"
    MODEL_PARAMETERS = "model_params"
    SESSION = "training_session"
    TRAINING_TIME = "run_time"


class TIMESERIES_KEYS:
    """Class of constants with labels for the keys used in the timeseries of the DkuGluonDataset class"""

    START = FieldName.START
    TARGET = FieldName.TARGET
    TARGET_NAME = "target_name"
    TIME_COLUMN_NAME = "time_column_name"
    FEAT_DYNAMIC_REAL = FieldName.FEAT_DYNAMIC_REAL
    FEAT_DYNAMIC_REAL_COLUMNS_NAMES = "feat_dynamic_real_columns_names"
    IDENTIFIERS = "identifiers"


class ROW_ORIGIN:
    """Class of constants to label if the row is a forecast, historical data, ..."""

    COLUMN_NAME = "row_origin"
    FORECAST = "forecast"
    HISTORY = "history"
    TRAIN = "train"
    EVALUATION = "evaluation"


EVALUATION_METRICS_DESCRIPTIONS = {
    "MSE": "Mean Squared Error",
    "MASE": "Mean Absolute Scaled Error",
    "MAPE": "Mean Absolute Percentage Error",
    "sMAPE": "Symmetric Mean Absolute Percentage Error",
    "MSIS": "Mean Scaled Interval Score",
    "RMSE": "Root Mean Square Error",
    "ND": "Normalized Deviation",
    "mean_wQuantileLoss": "Mean Weighted Quantile Loss",
}


METRICS_COLUMNS_DESCRIPTIONS = {
    METRICS_DATASET.MODEL_COLUMN: "Model name",
    METRICS_DATASET.MODEL_PARAMETERS: "Parameters used for training",
    METRICS_DATASET.SESSION: "Timestamp of training session",
    METRICS_DATASET.TARGET_COLUMN: "Aggregated and per-time-series metrics",
    METRICS_DATASET.TRAINING_TIME: "Time elapsed during model training and evaluation (in seconds)",
    ROW_ORIGIN.COLUMN_NAME: "Row origin",
}

# regex pattern to match the timestamps used for training sessions
TIMESTAMP_REGEX_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}Z"


FORECASTING_STYLE_PRESELECTED_MODELS = {
    "auto_univariate": ["trivial_identity", "seasonal_naive", "simplefeedforward"],
    "auto_performance_univariate": ["trivial_identity", "seasonal_naive", "simplefeedforward", "deepar", "transformer"],
    "auto_multivariate": ["trivial_identity", "seasonal_naive", "simplefeedforward"],
    "auto_performance_multivariate": [
        "trivial_identity",
        "seasonal_naive",
        "simplefeedforward",
        "deepar",
        "transformer",
    ],
}

CUSTOMISABLE_FREQUENCIES_OFFSETS = Week

DEFAULT_SEASONALITIES = {
    "min": 1,
    "H": 24,
    "D": 7,
    "W-SUN": 52,
    "W-MON": 52,
    "W-TUE": 52,
    "W-WED": 52,
    "W-THU": 52,
    "W-FRI": 52,
    "W-SAT": 52,
    "M": 12,
    "B": 5,
    "Q-DEC": 4,
}


class GPU_CONFIGURATION:
    NO_GPU = "no_gpu"
    CONTAINER_GPU = "container_gpu"
    CUDA_VERSION = "{CUDA_VERSION}"


# Training timeseries must be at least twice as long as the forecasting horizon (or some models will fail at training)
MINIMUM_TIMESERIES_LENGTH_TO_HORIZON_RATIO = 2
MINIMUM_FORECASTING_HORIZON = 1
