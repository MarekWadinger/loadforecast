# -*- coding: utf-8 -*-
# Copyright (c) 2021 Marek Wadinger

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from prophet.serialize import model_to_json
from prophet import Prophet
from loadforecast.warm_start import stan_init


class LoadProphet(Prophet):
    """Fit the Prophet model.

    Parameters
    ----------
    df: pd.DataFrame containing the history. Must have columns ds (date
        type) and y, the time series. If self.growth is 'logistic', then
        df must also have a column cap that specifies the capacity at
        each ds.
    country:
        Name of the country, like 'UnitedStates' or 'US'
    yearly_seasonality:
        Fit yearly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality:
        Fit weekly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality:
        Fit daily seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    seasonality_prior_scale:
        Parameter modulating the strength of the seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified for individual seasonalities using
        add_seasonality.
    holidays_prior_scale:
        Parameter modulating the strength of the holiday components model, unless overridden in the holidays input.
    changepoint_prior_scale:
        Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.

    Returns
    -------
    The fitted Prophet object.
    """

    def __init__(
        self,
        df=None,
        pretrained_model=None,
        country="BE",
        yearly_seasonality="auto",
        weekly_seasonality=28,
        daily_seasonality=True,
        seasonality_prior_scale=10,
        holidays_prior_scale=0.1,
        changepoint_prior_scale=0.001,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        holidays=None,
        seasonality_mode="additive",
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        stan_backend=None,
    ):
        super().__init__(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend,
        )
        if country:
            super().add_country_holidays(country)
        if df is not None and pretrained_model is not None:
            super().fit(df, init=stan_init(pretrained_model))
        elif df is not None and pretrained_model is None:
            super().fit(df)

    def prediction(self, prediction_periods=24 * 4, frequency="15min", floor_lim=0):
        """Predict using the prophet model.

        Parameters
        ----------
        prediction_periods:
            Int number of periods to forecast forward.
        frequency:
            Any valid frequency for pd.date_range, such as 'S', 'T', 'H', 'D' or 'M'.
        floor_lim:
            Lower limit of predicted values.

        Returns
        -------
        A pd.DataFrame with the forecast components.
        """
        # Extend df to the future by specified number of hours
        future = super().make_future_dataframe(
            periods=prediction_periods, freq=frequency
        )
        # Set lower bound on given prediction
        future["floor"] = floor_lim
        # Make prediction
        forecast = super().predict(future)
        return forecast

    def model_dump(self, name):
        with open(name, "w") as fout:
            json.dump(model_to_json(self), fout)  # Save model
        return self
