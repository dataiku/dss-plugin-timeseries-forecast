AVAILABLE_MODELS = ["naive", "simplefeedforward", "deepar", "transformer", "nbeats"]

EVALUATION_METRICS = ["MSE", "MASE", "MAPE", "sMAPE", "MSIS", "RMSE", "ND", "mean_wQuantileLoss"]


class METRICS_DATASET:
    TARGET_COLUMN = "target_column"
    MODEL_COLUMN = "model"
    AGGREGATED_ROW = "AGGREGATED"
