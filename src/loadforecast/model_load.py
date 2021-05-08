from collections import OrderedDict
import json

import numpy as np
import pandas as pd

from loadforecast.forecaster import LoadProphet

SIMPLE_ATTRIBUTES = [
    "growth",
    "n_changepoints",
    "specified_changepoints",
    "changepoint_range",
    "yearly_seasonality",
    "weekly_seasonality",
    "daily_seasonality",
    "seasonality_mode",
    "seasonality_prior_scale",
    "changepoint_prior_scale",
    "holidays_prior_scale",
    "mcmc_samples",
    "interval_width",
    "uncertainty_samples",
    "y_scale",
    "logistic_floor",
    "country_holidays",
    "component_modes",
]

PD_SERIES = ["changepoints", "history_dates", "train_holiday_names"]

PD_TIMESTAMP = ["start"]

PD_TIMEDELTA = ["t_scale"]

PD_DATAFRAME = ["holidays", "history", "train_component_cols"]

NP_ARRAY = ["changepoints_t"]

ORDEREDDICT = ["seasonalities", "extra_regressors"]


def model_load(model_json):
    """Deserialize a Prophet model from json string.

    Deserializes models that were serialized with model_to_json.

    Parameters
    ----------
    model_json: Serialized model string

    Returns
    -------
    Prophet model.
    """
    with open(model_json, "r") as fin:
        attr_dict = json.loads(json.load(fin))
    model = LoadProphet()  # We will overwrite all attributes set in init anyway
    # Simple types
    for attribute in SIMPLE_ATTRIBUTES:
        setattr(model, attribute, attr_dict[attribute])
    for attribute in PD_SERIES:
        if attr_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            s = pd.read_json(attr_dict[attribute], typ="series", orient="split")
            if s.name == "ds":
                if len(s) == 0:
                    s = pd.to_datetime(s)
                s = s.dt.tz_localize(None)
            setattr(model, attribute, s)
    for attribute in PD_TIMESTAMP:
        setattr(model, attribute, pd.Timestamp.utcfromtimestamp(attr_dict[attribute]))
    for attribute in PD_TIMEDELTA:
        setattr(model, attribute, pd.Timedelta(seconds=attr_dict[attribute]))
    for attribute in PD_DATAFRAME:
        if attr_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            df = pd.read_json(
                attr_dict[attribute], typ="frame", orient="table", convert_dates=["ds"]
            )
            if attribute == "train_component_cols":
                # Special handling because of named index column
                df.columns.name = "component"
                df.index.name = "col"
            setattr(model, attribute, df)
    for attribute in NP_ARRAY:
        setattr(model, attribute, np.array(attr_dict[attribute]))
    for attribute in ORDEREDDICT:
        key_list, unordered_dict = attr_dict[attribute]
        od = OrderedDict()
        for key in key_list:
            od[key] = unordered_dict[key]
        setattr(model, attribute, od)

    # Params (Dict[str, np.ndarray])
    model.params = {k: np.array(v) for k, v in attr_dict["params"].items()}
    # Skipped attributes
    model.stan_backend = None
    model.stan_fit = None
    return model
