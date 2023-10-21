import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from pmdarima.arima import ADFTest
from pykalman import KalmanFilter
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from main.utils.common import theil
from typing import Tuple, List


@dataclass
class DataAnalyzer:
    file_name: str
    delimiter: str = ";"

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load original data to DataFrame and prepare them for analysis"""

        df = pd.read_csv(f"{self.file_name}.csv", delimiter=self.delimiter)

        # transform the data to make it suitable for analysis
        df = df[df.columns[::-1]].T
        df.columns = ["GDP"]
        df = df.astype(float)

        # set the months as indexes
        df.index = pd.to_datetime(df.index, format="%b-%y")

        return df

    @staticmethod
    def train_test_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Split input DataFrame to train and test set."""

        train_size = int(len(df.values) * 0.8)
        return df[0:train_size], df[train_size:]

    @staticmethod
    def get_fitted_arima_model(
        train_data: pd.Series, order: Tuple[int, int, int]
    ) -> ARIMAResults:
        """Create and fit ARIMA model"""

        arima_model = ARIMA(train_data, order=order)
        return arima_model.fit()

    @staticmethod
    def get_fitted_sarimax_model(
        train_data: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
    ) -> ARIMAResults:
        """Create and fit SARIMAX model"""

        arima_model = sm.tsa.SARIMAX(
            train_data, order=order, seasonal_order=seasonal_order
        )
        return arima_model.fit()

    @staticmethod
    def apply_kalman_filter(
        observations: np.ndarray, datetime_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Apply the Kalman filter and return DataFrame with the filtered data."""

        # Define the Kalman filter model
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=observations[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.5,
        )

        # Apply the Kalman filter to the observed data
        filtered_state_means, filtered_state_covariances = kf.filter(observations)

        # Return DataFrame with the filtered data
        return pd.DataFrame(
            {"GDP": filtered_state_means.flatten()}, index=datetime_index
        )

    @staticmethod
    def print_adf_test_result(df: pd.DataFrame) -> None:
        """Calculates and prints ADF test results for DataFrame."""

        adf_test = ADFTest(alpha=0.05)
        print(adf_test.should_diff(df))

    @staticmethod
    def print_mape_theil_metrics(test_data: pd.Series, predictions: List) -> None:
        """Calculate and print mean absolute percentage error and theil."""

        mape = mean_absolute_percentage_error(test_data, pd.Series(predictions))
        print(f"MAPE:\n{mape}")
        print(f"Theil:\n{theil(test_data.values, predictions)}")

    @staticmethod
    def print_model_summary_and_metrics(
        model_fit: ARIMAResults, observed_data: pd.Series
    ) -> None:
        """Calculates and prints the summary statistics of an ARIMA model."""

        print(model_fit.summary())
        residuals = model_fit.resid

        print("Durbin-Watson:")
        print(durbin_watson(residuals))

        print("R2:")
        tss = np.sum(
            (observed_data - np.mean(observed_data)) ** 2
        )  # calculate the total sum of squares (TSS)
        rss = np.sum(residuals**2)  # calculate the residual sum of squares (RSS)
        r2 = 1 - rss / tss  # calculate R-squared
        print(r2)
