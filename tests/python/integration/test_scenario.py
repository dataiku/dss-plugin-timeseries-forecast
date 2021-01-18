import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")


test_kwargs = {
    "user": "user1",
    "project_key": "TIMESERIESTEST",
    "logger": logging.getLogger("dss-plugin-test.timeseries-forecast.test_scenario"),
}


def test_run_timeseries_forecast_regular(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="Regular", **test_kwargs)


def test_run_timeseries_forecast_partition(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="Partitions", **test_kwargs)


def test_run_timeseries_sql_forecast_regular(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="SQLRegular", **test_kwargs)


def test_run_timeseries_sql_forecast_partition(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="SQLPartitions", **test_kwargs)
