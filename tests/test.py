import pandas as pd

from loadforecast.forecaster import LoadProphet
from loadforecast.surpress_output import suppress_printout
from loadforecast.model_load import model_load

# Load csv file as pandas data frame
df = pd.read_csv('miris_load_15.csv')
# Rename dataframe columns so they match requirements
df_new = df[['DateTime', 'Load']].rename(columns={'DateTime': 'ds', 'Load': 'y'})
# Optional: Transformation from type str to datetime (handy when plotting using matplotlib
df_new['ds'] = pd.to_datetime(df_new['ds'])
# strip last day
df_new1 = df_new.loc[df_new['ds'] < '2019-06-20', :]

# Optional: Suppress print out of stdout and stderr upstream
with suppress_printout():
    # Initialize model fitted on provided data
    m = LoadProphet(df_new1)
    # Make future prediction
    forecast = m.prediction()

# Save model as json
m.model_dump('loadForecast_model.json')

# Load model from json
m1 = model_load('loadForecast_model.json')

# Make prediction on loaded model
forecast1 = m1.predict()

# Warm-starting the fit from the model parameters of the earlier model
m2 = LoadProphet(df_new, m1)

forecast2 = m2.predict()

# Return predictions in file
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('14_day_load_forecast.csv')
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json('14_day_load_forecast.json')
