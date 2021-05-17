# Loadforecast
A package for an easy time series forecasting of an electrical load based on facebook prophet.
It inherits the base of prophet which is additive model where non-linear trends are fit with yearly, weekly, 
and daily seasonality, with optional holiday effects. It is robust to missing data and outliers as well.

Package is build around child class of original prophet, replacing initial values of attributes by those tuned for 
electrical load forecasting using grid search method. Package contains functions that are mainly packing up original 
functions to more user-friendly ones.

We are sorry for inconvenience but currently it is necessary to install `prophet` package manually as main dependency.
For more information about the installation procedure check https://github.com/facebook/prophet

Loadforecast is on PyPI, so you can use `pip` to install it.
```bash
pip install loadforecast
```

**API Demo**:
```
# Initialize model on pandas.DataFrame containing columns 'DateTime' and 'Load'
m = LoadProphet(df)
```
```
# Make default one day prediction with sample period of 15 minutes.
forecast = m.prediction()
```
## Changelog
[Changelog File](CHANGELOG.md)

## License

Loadforecast is licensed under the [MIT license](LICENSE).
