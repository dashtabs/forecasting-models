import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from main.utils.common import theil
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# read data from csv
df = pd.read_csv('CPI-USA.csv', header=None)
df = df[df.columns[::-1]]
df = df.T
df.drop(df.tail(1).index, inplace=True)  # the names appear at the end of the list, we get rid of them
df = df.set_index([0])
df.index = pd.to_datetime(df.index, format='%b-%y')

df.columns = ['All items', 'Food', 'Cereals/Bakery', 'Meats/Eggs',
              'Dairy', 'Fruits/Veg', 'Beverages NA', 'OFaH',
              'Other foods', 'FAwfHome', 'Energy', 'EnCommodities', 'EnServices']

df = df.astype(float)
plt.figure(figsize=(15, 6))
plt.plot(df['All items'], label='CPI All data')
plt.legend()
plt.show()

# decompose the data
series = df['All items']
result = seasonal_decompose(series, model='additive')
result.plot()
plt.show()

# create a correlation matrix
X = df.drop('All items', axis=1)
y = df['All items']

plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor['All items'])
relevant_features = cor_target[cor_target < 0.8]
print(relevant_features)

exog = df[['Food', 'Cereals/Bakery', 'Meats/Eggs', 'Dairy', 'Fruits/Veg', 'Beverages NA', 'OFaH', 'Other foods',
          'FAwfHome', 'EnCommodities', 'EnServices']]

target = y.ewm(span=3, adjust=False).mean()

train_size = int(len(target) * 0.8)
train_y, test_y = target[0:train_size], target[train_size:]
train_X, test_X = exog[0:train_size], exog[train_size:]

# uncomment for heteroscedasticity test and change the first 1 to 0 in the trend component
# train_X = sm.add_constant(train_X)
# test_X = sm.add_constant(test_X)

mod = sm.tsa.statespace.SARIMAX(endog=train_y, exog=train_X, order=(3, 1, 0), trend=[1, 1, 1, 1])
res = mod.fit()
print(res.summary())

# Perform White's test for heteroscedasticity
# white_test = sm.stats.diagnostic.het_white(res.resid, res.model.exog)
#
# # Extract the test statistics and p-values
# test_statistic = white_test[0]
# p_value = white_test[1]

# # Print the test results
# print("White's Test Statistic:", test_statistic)
# print("p-value:", p_value)

print("Durbin-Watson:")
print(durbin_watson(res.resid))

pr = res.get_prediction(start='2016-08-01', end='2021-11-01', exog=train_X)
preds = pr.predicted_mean

ax = train_y.plot(figsize=(15, 5))
preds.plot(ax=ax)
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Train Data and Fitted Data')
plt.show()

# uncomment for many-steps-ahead forecast and comment lines 108-121
# fo = res.get_prediction(start='2021-12-01', end='2023-03-01', exog=test_X)
# forecasts = fo.predicted_mean

forecasts = []
history_endog = train_y.tolist()
history_exog = train_X.values.tolist()

dates = ['2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01']

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
print(theil(test_y, forecasts))

plt.plot(test_y.index, test_y.values, label='Test')
plt.plot(test_y.index, forecasts, label='Test Forecast')

plt.xlabel('Date')
plt.ylabel('CPI')
plt.title('Forecast')
plt.legend()
plt.show()

plt.plot(train_y.index, train_y.values, label='Train')
plt.plot(test_y.index, test_y.values, label='Test')
plt.plot(train_y.index, preds, label='ARIMA Train Forecast')
plt.plot(test_y.index, forecasts, label='ARIMA Test Forecast')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

