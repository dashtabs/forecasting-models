# Forecasting models 
The project covers the analysis and forecasting of individual macroeconomic indicators based on several preliminary data using statistical models (ARMA, ARIMA, AR+trend), fuzzy rules, and LSTM.
Real data were used upon request to the financial information provider Bloomberg for analysis and forecasting. All data are monthly since August 2016.

## Input data requirements

All data files are `.csv`. All data is a monthly time series. In the .csv files data is presented in rows with the most current data to the left and old data -- to the right. The first row in the tables is always the month in format *Mmm-yy*. This row is converted to a datetime index in the program.

 - `gdp.csv` – one-dimensional dataset.
 - `gdp_multi.csv` -- two-dimensional dataset to forecast real value againstvthe nominal one.
 - `CPI-USA.csv` -- many-dimensional dataset to forecast CPI index values using many
   features after performing correlation analysis and removing irrelevant indexes.
 - `PPO.csv` -- two-dimensional dataset.

## Output examples
### Data pre-processing 
Time Series Decomposition| Autocorrelation plot | Correlation Matrix
 ------------------- |  ---------------------- | ----------------
![image](https://imgur.com/gpbGMsr.png) | ![image](https://i.imgur.com/WbZwzHu.png) | ![Imgur](https://i.imgur.com/VW0LLv7.png)

### Forecasts 
LSTM model | AR(2) + trend model | Fuzzy rule-based forecast
----------- | -------------- | ------------
![Imgur](https://i.imgur.com/6SDNFso.png) |![Imgur](https://i.imgur.com/csSbLOk.png) | ![Imgur](https://i.imgur.com/C8wckl9.png)

### AR(2)+trend model metrics

AIC | DW | R2 | MAPE | Theil
-----| -----| -----| --------| -------
-149.996 | 1.865 | 0.9034 | 0.1866 | 0.0874

> Theoretical background coming soon.

## Project Structure
### Utils
- `analyzer.py` – contains a class for GDP data pre-processing; performs ADF-test, prints metrics.

-  `common.py` – calculates Theil coefficient.

-  `visualizer.py` – visualizes autocorrelation and partial autocorrelation plots, also time series and time series decomposition.
### Other files

-  `cpi_us_trend.py` – builds multi-step and single-step forecasting models for CPI data.

-  `gdp_us_auto.py` – implements an approach using auto_arima.

-  `gdp_us_empiric` – implements an approach with manual setting of model parameters.

-  `gdp_us_fuzzy.py` – implements fuzzy time series models.

-  `gdp_us_lstm.py` – implements the construction of the model using the LSTM.

-  `gdp_us_other.py` – implements a model with a step-by-step forecast with model retraining.

- `ppo_ua.py` – implements the construction of a model and forecast for PPO-data.



