# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


TEST_PROJECT_KEY = "TIMESERIESTEST"


def test_run_timeseries_forecast_regular(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="Regular")


def test_run_timeseries_forecast_partition(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="Partitions")


def test_run_timeseries_sql_forecast_regular(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="SQLRegular")


def test_run_timeseries_sql_forecast_partition(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="SQLPartitions")
