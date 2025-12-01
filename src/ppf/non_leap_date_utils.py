from datetime import date, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np

# Constants for skipped leap day
MONTH_TO_REMOVE = 2
DAY_TO_REMOVE = 29

def subtract_non_leap_days(
    date_to_subtract_from: date,
    days_to_subtract: int,
) -> date:
    """
    Subtract a number of calendar days from a given date,
    skipping February 29 (leap day) when encountered.

    This is useful for time series applications where uniformity across
    years is required by removing variability caused by leap years.

    Parameters
    ----------
    date_to_subtract_from : date
        The starting reference date.

    days_to_subtract : int
        Number of valid (non-leap-day) days to subtract.

    Returns
    -------
    date
        The resulting date after subtracting the desired number of days,
        excluding leap days (Feb 29).
    """
    current_date = date_to_subtract_from
    count = 0
    while count < days_to_subtract:
        current_date -= timedelta(days=1)
        # Skip Feb 29 (leap day)
        if not (current_date.month == MONTH_TO_REMOVE and current_date.day == DAY_TO_REMOVE):
            count += 1
    return current_date


def add_non_leap_days(
    date_to_add_to: date,
    days_to_add: int,
) -> date:
    """
    Add a number of calendar days to a given date,
    skipping February 29 (leap day) when encountered.

    Useful for generating consistent time steps across years
    without including variable-length years.

    Parameters
    ----------
    date_to_add_to : date
        The starting reference date.

    days_to_add : int
        Number of valid (non-leap-day) days to add.

    Returns
    -------
    date
        The resulting date after adding the desired number of days,
        excluding leap days (Feb 29).
    """
    current_date = date_to_add_to
    count = 0
    while count < days_to_add:
        current_date += timedelta(days=1)
        # Skip Feb 29 (leap day)
        if not (current_date.month == MONTH_TO_REMOVE and current_date.day == DAY_TO_REMOVE):
            count += 1
    return current_date


def calculate_non_leap_offset_dates(
    start_year: int,
    end_year: int,
    input_size: int,
    train_size: int,
    offset_days: int,
    horizon_size: int,
) -> Tuple[date, date]:
    """
    Calculate the adjusted start and end dates for a time series modeling window,
    avoiding Feb 29 (leap day) to ensure consistent daily intervals across years.

    This function accounts for lookback (input), training horizon, and offset
    and guarantees that no leap days are included in the calculation.

    Parameters
    ----------
    start_year : int
        Year corresponding to the start of the modeling range.

    end_year : int
        Year corresponding to the end of the modeling range.

    input_size : int
        Number of past time steps used for model input.

    train_size : int
        Number of samples used for training before forecasting.

    offset_days : int
        Days to shift the window forward or backward (can be negative).

    horizon_size : int
        Number of steps in the prediction horizon.

    Returns
    -------
    Tuple[date, date]
        A tuple containing the adjusted (start_date, end_date) after offset
        and removal of leap days.

    Notes
    -----
    This function ensures enough buffer around start and end dates to apply
    offsetting and avoid leap-day inconsistencies in training/testing windows.
    """
    # Define the raw start and end dates
    orig_start = date(start_year, 1, 1)
    orig_end = date(end_year, 12, 31)

    # Compute total days to shift start date (lookback + training + offset + horizon)
    offset_days_start = input_size + train_size + max(0, -offset_days) + (horizon_size - 1)
    
    # Compute total days to shift end date (offset + horizon)
    offset_days_end = max(offset_days, 0) + (horizon_size - 1)

    # Apply leap-day-skipping subtraction and addition
    new_start = subtract_non_leap_days(orig_start, offset_days_start)
    new_end = add_non_leap_days(orig_end, offset_days_end)

    return new_start, new_end