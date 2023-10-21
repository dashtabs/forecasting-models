import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from main.utils.common import theil
from sklearn.metrics import mean_absolute_percentage_error, r2_score

df = pd.read_csv('gdp_multi.csv', header=None)
df = df[df.columns[::-1]]
df = df.T
df.drop(df.tail(1).index, inplace=True)  # the names appear at the end of the list, we get rid of them
df = df.set_index([0])
df.index = pd.to_datetime(df.index, format='%b-%y')

df.columns = ['GDP real', 'GDP nominal']
df = df.astype(float)

plt.figure(figsize=(15, 6))
plt.plot(df['GDP real'], label='GDP real')
plt.legend()
plt.show()

series = df['GDP real']
# result = seasonal_decompose(series, period=12, model='additive')
# result.plot()
# plt.show()
train_y1, test_y1 = series[0:int(len(series) * 0.8)], series[int(len(series) * 0.8):]

y = df['GDP real']
# target = y.ewm(span=10, adjust=False).mean()
target = y
exog = df['GDP nominal']

train_size = int(len(target) * 0.8)
train_y, test_y = target[0:train_size], target[train_size:]
train_X, test_X = exog[0:train_size], exog[train_size:]

mod = sm.tsa.statespace.SARIMAX(endog=train_y, exog=train_X, order=(2, 2, 0), trend=[1, 1, 1, 1])

res = mod.fit()
print(res.summary())

print("Durbin-Watson:")
print(durbin_watson(res.resid))

pr = res.get_prediction(start='2016-08-01', exog=train_X)
preds = pr.predicted_mean

ax = train_y.plot(figsize=(15, 5))
preds.plot(ax=ax)
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Train Data and Fitted Data')
plt.show()

# model = sm.tsa.statespace.SARIMAX(endog=train_y, exog=train_X, order=(2, 1, 0), trend=[1, 1, 1, 1])
# model_fit = model.fit()
# forecasts = model_fit.get_prediction(start=len(train_y), end=(len(train_y)+len(test_y)-1), exog=test_X)
# forecasts = forecasts.predicted_mean
forecasts = []
history_endog = train_y.tolist()
history_exog = train_X.values.tolist()

dates = ['2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
         '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01', '2022-01-01']


for t in range(len(test_y)):
    model = sm.tsa.statespace.SARIMAX(endog=history_endog, exog=history_exog, order=(3, 1, 0), trend=[1, 1, 1, 1])
    model_fit = model.fit()
    forecast = model_fit.get_prediction(start=len(history_endog), end=len(history_endog), exog=[test_X.iloc[t]])
    forecasts.append(forecast.predicted_mean[0])
    print(f"TestY: {test_y.iloc[t]}")
    history_endog.append(test_y.iloc[t])
    history_exog.append(test_X.iloc[t])

mape = mean_absolute_percentage_error(test_y, forecasts)
print("MAPE:")
print(mape)
print("Theil:")
print(theil(test_y1, forecasts))

plt.plot(test_y1.index, test_y1.values, label='Test')
plt.plot(test_y1.index, forecasts, label='Test Forecast')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.title('Forecast')
plt.legend()
plt.show()

plt.plot(train_y.index, train_y.values, label='Train')
plt.plot(test_y.index, test_y.values, label='Test')
plt.plot(train_y.index, preds, label='ARIMA Train Forecast')
plt.plot(test_y.index, forecasts, label='ARIMA Test Forecast')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.title('AR(2)+features Forecast')
plt.legend()
plt.show()

r2 = r2_score(train_y, preds)
print(r2)