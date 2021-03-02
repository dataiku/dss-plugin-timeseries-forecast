# Time Series Forecast Plugin

![Build status](https://github.com/dataiku/dss-plugin-timeseries-forecast/actions/workflows/auto-make.yml/badge.svg) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/dataiku/dss-plugin-timeseries-forecast?logo=github) ![Support level](https://img.shields.io/badge/support-Tier%202-yellowgreen)

This Dataiku DSS plugin provides recipes to forecast multivariate time series from year to minute frequency with Deep Learning and statistical models.

Documentation: https://www.dataiku.com/product/plugins/timeseries-forecast/

## Release notes

**Version 1.0.0 (2021-02)** - Initial release

- 🔥 Multivariate forecasting, scaling to 1000s of time series
- 😎 Deep Learning models from GluonTS (DeepAR, Transformer, ...)
- 🗓 Frequencies from year to minute, inc. business day
- 🐍 Easy-to-install code environment in Python
- 👾 GPU version to train your models even faster

**Version 1.0.1 (2021-03)** - Bugfix release

- 🪲 Bugfix with the FeedForward model when using external features of different lengths
- 💅 Small naming and logging enhancements

## License

This plugin is distributed under the [Apache License version 2.0](LICENSE).
