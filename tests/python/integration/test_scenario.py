import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")
logger = logging.getLogger("dss-plugin-test.timeseries-forecast.test_scenario")


def test_run_timeseries_forecast_regular(user_clients):
    dss_scenario.run("default", user_clients["default"], "TIMESERIESTEST", "Regular", logger)


def test_run_timeseries_forecast_partition(user_clients):
    dss_scenario.run("default", user_clients["default"], "TIMESERIESTEST", "Partitions", logger)
