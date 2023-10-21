import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from dataclasses import dataclass
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, List, Union



@dataclass
class DataVisualizer:
    figsize: Tuple = (15, 5)
    ax: Any = None

    def _plot(self, title: str) -> None:
        self.ax.set_title(title)
        plt.show()

    def visualize_data(
        self,
        df: Union[pd.DataFrame, Series],
        *dataframes: Union[pd.DataFrame, Series],
        legend: List[str],
        title: str,
    ) -> None:
        """Visualize DataFrame data."""

        self.ax = df.plot(figsize=self.figsize)

        for data in dataframes:
            data.plot(ax=self.ax)

        plt.legend(legend)
        self._plot(title)

    def visualize_acf_pacf(self, df: pd.DataFrame, title: str = "Data") -> None:
        """Visualize the ACF and PACF plots."""

        self.ax = df.plot(figsize=self.figsize)

        plot_acf(df)
        self._plot(f"ACF {title}")

        plot_pacf(df)
        self._plot(f"PACF {title}")

    def visualize_seasonal_decompose(self, series: Series) -> None:
        """Perform seasonal decomposition using moving averages and visualize results."""

        result = seasonal_decompose(series, model="additive")

        self.ax = series.plot(figsize=self.figsize)
        self.ax.set_title("Series decomposition")
        result.plot()
        self._plot("Seasonal Decomposition")
