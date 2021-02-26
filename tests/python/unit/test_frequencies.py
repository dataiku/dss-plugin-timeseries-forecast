from gluonts.dataset.common import ListDataset
from gluonts_forecasts.model import Model
from gluonts_forecasts.trained_model import TrainedModel
from dku_constants import TIMESERIES_KEYS
import pandas as pd
import numpy as np


def test_minute_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-15 12:40:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "20min"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-15 13:00:00")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-01-15 13:20:00")


def test_hours_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-15 12:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "6H"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-15 18:00:00")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-01-16 00:00:00")


def test_day_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-15 00:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "3D"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-18")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-01-21")


def test_business_day_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-14 00:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "B"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-15")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-01-18")


def test_week_sunday_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-17 00:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "W-SUN"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-24")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-01-31")


def test_week_tuesday_frequency():
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-19 00:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "W-TUE"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-01-26")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-02-02")


def test_month_frequency():
    """This test covers all month frequencies (quarter=3M, semester=6M, year=12M) """
    prediction_length = 1
    timeseries = {
        TIMESERIES_KEYS.START: "2021-01-31 00:00:00",
        TIMESERIES_KEYS.TARGET: np.array([12, 13]),
        TIMESERIES_KEYS.TARGET_NAME: "target",
        TIMESERIES_KEYS.TIME_COLUMN_NAME: "date",
    }
    frequency = "4M"
    gluon_dataset = ListDataset([timeseries], freq=frequency)
    model = Model(
        "simplefeedforward",
        model_parameters={"activated": True, "kwargs": {}},
        frequency=frequency,
        prediction_length=prediction_length,
        epoch=1,
        batch_size=8,
        num_batches_per_epoch=5,
    )
    evaluation_forecasts_df = model.train_evaluate(gluon_dataset, gluon_dataset, make_forecasts=True, retrain=True)[2]
    assert evaluation_forecasts_df["index"].iloc[0] == pd.Timestamp("2021-05-31")

    trained_model = TrainedModel(
        model_name="simplefeedforward",
        predictor=model.predictor,
        gluon_dataset=gluon_dataset,
        prediction_length=prediction_length,
        quantiles=[0.5],
        include_history=True,
    )
    trained_model.predict()
    forecasts_df = trained_model.get_forecasts_df(session="2021-01-01")
    assert forecasts_df["date"].iloc[0] == pd.Timestamp("2021-09-30")
