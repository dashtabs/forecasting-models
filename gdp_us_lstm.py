import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from main.utils.common import theil

df = pd.read_csv('gdp.csv', delimiter=';')

df = df[df.columns[::-1]].T
df.columns = ['GDP']
df = df.astype(float)
# set the months as indexes
df.index = pd.to_datetime(df.index, format='%b-%y')
data_array = df.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_array.reshape(-1, 1))
# Divide data into training and testing sets
train_size = int(len(scaled_data) * 0.75)
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:, :]
test_data_index = data_array[train_size:].index

def create_dataset(dataset, window_size=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - window_size - 1):
        data_x.append(dataset[i:(i + window_size), 0])
        data_y.append(dataset[i + window_size, 0])
    return np.array(data_x), np.array(data_y)

# Generate time series dataset for LSTM
window_size = 5
train_x, train_y = create_dataset(train_data, window_size)
test_x, test_y = create_dataset(test_data, window_size)
print(len(test_x))

# Reshape input to [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
print(len(test_x))
# Construct LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(1, window_size)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='poisson', optimizer='adam')

# Train LSTM model
model.fit(train_x, train_y, epochs=1200, batch_size=1, verbose=0)

train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Revert predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])
# Calculate the R2 coefficient
# print(train_predict)
r2 = r2_score(train_y[0], train_predict)
# Print the R2 coefficient
print("R2 Coefficient:", r2)

mape = mean_absolute_percentage_error(test_y[0], test_predict[:, 0])
print("MAPE:")
print(mape)
print("Theil:")
print(theil(test_y[0], test_predict[:, 0]))

plt.figure(figsize=(12, 6))
# plt.plot(df[:-len(test_data_index)], label="Training data GDP")
plt.plot(df['GDP'][:-4], label="Actual GDP")
plt.plot(df['GDP'].index[5:len(train_predict)+5], train_predict, label='LSTM Train Forecast')
plt.plot(test_data_index[2:-4], test_predict, label="LSTM Test Forecast")
plt.xlabel("Date")
plt.ylabel("GDP")
plt.title(f" GDP Prediction using LSTM")
plt.legend()
plt.show()

r2 = r2_score(df['GDP'][-18:-4], test_predict[:, 0])

