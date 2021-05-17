# 0.0.4 (2021-05-17)
Updated model loading, so now it is possible to input model unpacked dictionary.
Made country an optional parameter not giving an error when empty.

# 0.0.3 (2021-05-02)

Solving collision between name-space of loadforecast and prophet which resulted in error
message during cross-validation of model.

# 0.0.2 (2021-04-29)

Minor changes in aesthetics.

# 0.0.1 (2021-04-28)

LoadForecast package that makes usage of Facebook Prophet even more convenient. 
LoadForecast overrides Prophet initial conditions with the ones suitable for 
electrical load forecasting.