AVAILABLE_MODELS = ["naive", "simplefeedforward", "deepar", "transformer", "nbeats"]

EVALUATION_METRICS = ["MSE", "MASE", "MAPE", "sMAPE", "MSIS", "RMSE", "ND", "mean_wQuantileLoss"]

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


class METRICS_DATASET:
    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"
