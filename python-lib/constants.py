class METRICS_DATASET:
    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"
    MODEL_PARAMETERS = "model_params"
    SESSION = "session"


AVAILABLE_MODELS = ["naive", "simplefeedforward", "deepar", "transformer", "nbeats"]

EVALUATION_METRICS = ["MSE", "MASE", "MAPE", "sMAPE", "MSIS", "RMSE", "ND", "mean_wQuantileLoss"]

EVALUATION_METRICS_DESCRIPTIONS = {
    "MSE": "Mean Squared Error",
    "MASE": "Mean Absolute Scaled Error",
    "MAPE": "Mean Absolute Percentage Error",
    "sMAPE": "Symmetric Mean Absolute Percentage Error",
    "MSIS": "",
    "RMSE": "Root Mean Square Error",
    "ND": "sum(abs_error)/sum(abs_target_sum)",
    "mean_wQuantileLoss": "Mean Weight Quantile Loss",
    METRICS_DATASET.MODEL_PARAMETERS: "Parameters used for training",
    METRICS_DATASET.SESSION: "Timestamp of training session",
}
