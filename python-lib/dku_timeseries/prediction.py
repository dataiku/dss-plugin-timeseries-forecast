import pandas as pd

class Prediction():
    def __init__(self, predictor, forecasting_horizon, confidence_intervals):
        self.predictor = predictor
        self.forecasting_horizon = forecasting_horizon
        self.confidence_intervals = confidence_intervals

    def predict(self, context_dataset):
        forecasts = self.predictor.predict(context_dataset)
        sample_forecasts = list(forecasts)[0]

        columns = ["forecasts"]
        series = [sample_forecasts.mean_ts]
        for conf_int in self.confidence_intervals:
            columns += ["quantile_{}".format(conf_int)]
            series += [sample_forecasts.quantile_ts(conf_int)]

        df = pd.concat(series, axis=1).reset_index()
        df.columns = ["time"] + columns

        return 

    def get_results_dataframe(self):
        return
