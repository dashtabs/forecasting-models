import pandas as pd
from pmdarima import auto_arima
from utils.analyzer import DataAnalyzer
from utils.visualizer import DataVisualizer

analyzer = DataAnalyzer(file_name="gdp")
visualizer = DataVisualizer()

# reading time series data from csv file and prepare for analysis
original_df = analyzer.load_and_prepare_data()

print("ORIGINAL DATA - HEAD")
print(original_df.head())

# plot original data
visualizer.visualize_data(df=original_df, legend=["GDP Data"], title="GDP Data")

# split our data to train and test set
train, test = analyzer.train_test_split(df=original_df)

stepwise_model = auto_arima(
    train,
    start_p=1,
    start_q=1,
    max_p=5,
    max_q=5,
    m=12,
    start_P=1,
    max_P=5,
    seasonal=True,
    std=1,
    start_Q=1,
    max_Q=5,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
print(stepwise_model.aic())


model = analyzer.get_fitted_sarimax_model(
    train_data=train, order=(2, 0, 0), seasonal_order=(0, 0, 0, 12)
)

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

# and make forecasts on the train set
forecasts = model.forecast(len(test))

# print mape and theil metrics
analyzer.print_mape_theil_metrics(test_data=test, predictions=forecasts)

visualizer.visualize_data(
    test,
    pd.Series(forecasts),
    legend=["Test Data", "Forecasts"],
    title="Test Data and Forecasts",
)
