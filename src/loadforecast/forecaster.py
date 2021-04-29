# -*- coding: utf-8 -*-
# Copyright (c) 2021 Marek Wadinger

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from prophet.serialize import model_to_json
from prophet import Prophet


class LoadProphet(Prophet):
    """Fit the Prophet model.

    Parameters
    ----------
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
            country='BE',
            yearly_seasonality='auto',
            weekly_seasonality=28,
            daily_seasonality=True,
            seasonality_prior_scale=10,
            holidays_prior_scale=0.1,
            changepoint_prior_scale=0.001
    ):
        super().__init__(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale
        )
        super().add_country_holidays(country)

    def predict(self, prediction_periods=24 * 4, frequency='15min', floor_lim=0):
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
        future = super().make_future_dataframe(periods=prediction_periods, freq=frequency)
        # Set lower bound on given prediction
        future['floor'] = floor_lim
        # Make prediction
        forecast = super().predict(future)
        return forecast

    def model_to_json(self, name):
        with open(name, 'w') as fout:
            json.dump(model_to_json(self), fout)  # Save model
        return self
