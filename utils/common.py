import math


def theil(actual: list, forecast: list) -> float:
    """Calculate Theil coefficient for a forecasting quality analysis."""

    s = len(forecast)
    numerator = math.sqrt(sum((a - f) ** 2 for a, f in zip(actual, forecast)) / s)
    denominator = math.sqrt(sum(a**2 for a in actual) / s) + math.sqrt(
        sum(f**2 for f in forecast) / s
    )
    theil_u = numerator / denominator

    return theil_u
