import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")


test_kwargs = {
    "user": "data_scientist_1",
    "project_key": "TIMESERIESTEST"
}


def test_run_timeseries_forecast_regular(dss_user_clients):
    dss_scenario.run(dss_user_clients, scenario_id="Regular", **test_kwargs)


def test_run_timeseries_forecast_partition(dss_user_clients):
    dss_scenario.run(dss_user_clients, scenario_id="Partitions", **test_kwargs)
