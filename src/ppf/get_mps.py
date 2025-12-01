from typing import Tuple, Optional
import pandas as pd

def get_mps_by_percentage(
    series: pd.Series,
    start_pct: float,
    end_pct: float
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Identify the Main Pollen Season (MPS) by locating the date range during which
    the cumulative pollen reach between the specified start and end percentages 
    of the total pollen.

    This function is useful for identifying the core pollen period within a season or year
    based on the cumulative pollen curve.

    Parameters
    ----------
    series : pd.Series
        A time series with a datetime index and numerical values (e.g., daily pollen).

    start_pct : float
        Lower cumulative threshold, expressed as a fraction (e.g., 0.15 for 15% of total).

    end_pct : float
        Upper cumulative threshold, expressed as a fraction (e.g., 0.85 for 85% of total).

    Returns
    -------
    Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
        A tuple containing:
        - The first date when the cumulative sum reaches or exceeds `start_pct * total`.
        - The first date when the cumulative sum reaches or exceeds `end_pct * total`.

        If the input series is empty, contains only zeros, or thresholds are not reached,
        the function returns `(None, None)` or partial results if only one threshold is met.

    Raises
    ------
    ValueError
        If `start_pct` or `end_pct` are not between 0 and 1.
        If `start_pct` > `end_pct`.
        If `series.index` is not datetime
        
    TypeError
        If `series` is not a pandas Series
    """
    # ----------------- Input Validation ----------------- #
    if not isinstance(series, pd.Series):
        raise TypeError("`series` must be a pandas Series.")

    if series.empty:
        return None, None

    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("`series` must have a datetime-like index.")

    if not (0 <= start_pct <= 1) or not (0 <= end_pct <= 1):
        raise ValueError("`start_pct` and `end_pct` must be between 0 and 1.")

    if start_pct > end_pct:
        raise ValueError("`start_pct` must be less than or equal to `end_pct`.")

    total = series.sum()
    if total == 0:
        return None, None

    # ----------------- Core Logic ----------------- #
    start_threshold = total * start_pct
    end_threshold = total * end_pct

    cum_sum = series.cumsum()

    # Find the earliest dates where cumulative pollen reach each threshold
    start_candidates = cum_sum[cum_sum >= start_threshold]
    end_candidates = cum_sum[cum_sum >= end_threshold]

    start_date = start_candidates.index[0] if not start_candidates.empty else None
    end_date = end_candidates.index[0] if not end_candidates.empty else None

    return start_date, end_date

def get_mps_by_threshold(
    series: pd.Series,
    threshold: float,
    consecutive_days: int
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Identify theMain Pollen Season (MPS) in a time series based on a value threshold 
    and a minimum number of consecutive days meeting or exceeding that threshold.

    This function finds the first and last periods in which the values of the series are 
    above or equal to a specified threshold for at least a given number of consecutive days.

    Parameters
    ----------
    series : pd.Series
        A time series indexed by datetime, containing numeric values.
        
    threshold : float
        The minimum value required for each day to be considered part of the MPS period.
        
    consecutive_days : int
        The minimum number of consecutive days that must meet or exceed the threshold 
        to qualify as a valid MPS period.

    Returns
    -------
    Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
        A tuple containing:
        - The start date of the first qualifying period (`pd.Timestamp`)
        - The end date of the last qualifying period (`pd.Timestamp`)
        
        If no valid period is found in either direction, returns `None` for that value.

    Raises
    ------
    TypeError:
        If `series` is not a pandas Series.
        If `threshold` is not a float or int.
        
    ValueError:
        If `series` is empty.
        If the series index is not datetime-like.
        If `consecutive_days` is not a positive integer.
    """
    # -------- Input Validation -------- #
    if not isinstance(series, pd.Series):
        raise TypeError("`series` must be a pandas Series.")

    if series.empty:
        raise ValueError("`series` must not be empty.")

    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("`series` index must be datetime-like (DatetimeIndex).")

    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise TypeError("`threshold` must be an non negative integer or float.")

    if not isinstance(consecutive_days, int) or consecutive_days <= 0:
        raise ValueError("`consecutive_days` must be a positive integer.")

    # -------- Core Logic -------- #
    # Create a boolean array: True where series meets or exceeds the threshold
    condition = series >= threshold
    indices = condition.index
    condition_values = condition.values

    def find_sequence(values: list[bool], direction: str = 'forward') -> Optional[pd.Timestamp]:
        """
        Helper function to locate the start or end of a run of `consecutive_days` True values.
        """
        run_length = 0
        index_range = range(len(values)) if direction == 'forward' else reversed(range(len(values)))

        for i in index_range:
            if values[i]:
                run_length += 1
                if run_length == consecutive_days:
                    if direction == 'forward':
                        # Return start of first valid run
                        return indices[i - consecutive_days + 1]
                    else:
                        # Return end of last valid run
                        return indices[i + consecutive_days - 1]
            else:
                run_length = 0

        return None

    # Find start and end of the MPS period
    start_date = find_sequence(condition_values, direction='forward')
    end_date = find_sequence(condition_values, direction='backward')

    return start_date, end_date