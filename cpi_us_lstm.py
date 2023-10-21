import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assume you have already prepared your data as numpy arrays: X_train, y_train, X_test, y_test
data = pd.read_csv('gdp.csv', delimiter=';')

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define window size and number of features
window_size = 10
num_features = data.shape[1]

# Create input-output sequences
X = []
y = []

for i in range(len(data) - window_size):
    window = scaled_data[i:i+window_size, :]  # Use all features in the window
    X.append(window)
    y.append(scaled_data[i+window_size, :])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, num_features)))  # Adjust the number of units (64) as needed
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
mse = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)