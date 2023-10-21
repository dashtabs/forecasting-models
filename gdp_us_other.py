import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from utils.analyzer import DataAnalyzer
from utils.visualizer import DataVisualizer

analyzer = DataAnalyzer(file_name="gdp")
visualizer = DataVisualizer()

# reading time series data from csv file and prepare for analysis
original_df = analyzer.load_and_prepare_data()
df = original_df.copy()
print("ORIGINAL DATA - HEAD")
print(original_df.head())

# plot original data
visualizer.visualize_data(df=original_df, legend=["GDP Data"], title="GDP Data")

# perform the Augmented Dickey–Fuller test (check for stability)
analyzer.print_adf_test_result(df=original_df)

# visualise the ACF and PACF plots
visualizer.visualize_acf_pacf(df=original_df)

# from both ADF-test and the plots we can see that the data is not stationary
# finally decompose data to trend, seasonality, and residuals
visualizer.visualize_seasonal_decompose(series=original_df["GDP"])

# we can compare how different differencing improves stationarity
analyzer.print_adf_test_result(df=original_df.diff().dropna())

# we need to apply a transformation to help us achieve stationarity
original_df["GDP"] = original_df["GDP"].ewm(span=5, adjust=False).mean()

# Create a new DataFrame with the filtered data after applying Kalman filter
filtered_df = analyzer.apply_kalman_filter(
    observations=original_df["GDP"].values, datetime_index=original_df.index
)

print("After the transformation")
analyzer.print_adf_test_result(df=filtered_df)

# plot transformed data
visualizer.visualize_data(
    df=filtered_df, legend=["Transformed GDP Data"], title="GDP Data"
)

# plot ACF and PACF again
visualizer.visualize_acf_pacf(df=filtered_df, title="Filtered Data")

# Now we can choose (p, d, q) and build a model
# Since the first lag is the most significant, we choose p = 1
# q = 2 from the ACF plot

# finally we split our data to train and test set
train, test = analyzer.train_test_split(df=filtered_df)

# setup ARIMA model
model = analyzer.get_fitted_arima_model(train_data=train, order=(1, 0, 2))

# print necessary metrics for model
analyzer.print_model_summary_and_metrics(model_fit=model, observed_data=train)

# we compare our train data with the fitted values
fitted = model.predict()
visualizer.visualize_data(
    original_df,
    fitted,
    legend=["Raw Data", "Fitted Data"],
    title="Raw Data and Fitted Data",
)

# and make forecasts on the test set
forecasts = []
history = train["GDP"].tolist()

for t in range(len(test)):
    model = ARIMA(history, order=(1, 0, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=2)
    forecasts.append(forecast[0])
    history.append(test.iloc[t]["GDP"])

# print mape and theil metrics
analyzer.print_mape_theil_metrics(test_data=test, predictions=forecasts)

plt.figure(figsize=(12, 6))

# plt.plot(train.index, train.values, label="Train")
# plt.plot(test.index, test.values, label="Test")
plt.plot(df.index, df.values, label="Original data")
plt.plot(train.index, fitted, label="ARIMA Train Forecast")
plt.plot(test.index, forecasts, label="ARIMA Test Forecast")

plt.xlabel("Date")
plt.ylabel("GDP")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()

