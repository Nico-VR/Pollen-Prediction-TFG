import os
import pandas as pd
import numpy as np
from typing import List


def save_predictions(
    model_name: str,
    predictions: pd.DataFrame,
    y_real: pd.Series,
    input_size: int,
    train_size: int,
    horizon_size: int,
    uses_covariates: bool,
    predictions_dir: str,
) -> None:
    """
    Save rolling forecast predictions to a CSV file in long-format,
    reconstructing forecasts for each target date from overlapping forecast windows.

    The output CSV contains:
    - `date`: target date of the forecast.
    - `pred_1`, ..., `pred_H`: forecast values for steps 1 to H in the horizon.
    - `real`: the observed value for that date, if available.

    This reconstruction handles partial forecasts at the beginning and end of the series
    where the full horizon is not available.

    Parameters
    ----------
    model_name : str
        Identifier for the forecasting model (used in filename).

    predictions : pd.DataFrame
        Rolling forecast results in wide format.
        Rows correspond to cutoff dates (forecast origins).
        Columns must be prediction steps named as either "1-pred", ..., "H-pred" or "pred_1", ..., "pred_H".

    y_real : pd.Series
        Series of observed values indexed by date.

    input_size : int
        Number of past observations used as input to the model (context window length).

    train_size : int
        Number of observations used for initial training (used in filename).

    horizon_size : int
        Forecast horizon length (number of steps predicted).

    uses_covariates : bool
        Whether the model used exogenous variables (affects filename suffix).

    predictions_dir : str
        Directory path to save the output CSV file.

    Returns
    -------
    None
        The function saves the reconstructed forecast DataFrame as a CSV file.

    Raises
    ------
    FileNotFoundError
        If `predictions_dir` does not exist.

    IOError
        If saving the CSV file fails for any reason.
    """
    if not os.path.isdir(predictions_dir):
        raise FileNotFoundError(f"The output path for predictions does not exist: {predictions_dir}")

    predictions = predictions.copy()
    predictions.index.name = "cutoff"

    records = []

    # --- Partial predictions for the initial target dates (less than full horizon) ---
    for i in range(1, horizon_size):
        record = {}
        cutoff_date = predictions.index[0]
        target_date = cutoff_date + pd.Timedelta(days=i)
        record["date"] = target_date

        # Initialize all prediction steps as NaN
        for step in range(1, horizon_size + 1):
            record[f"pred_{step}"] = np.nan

        # Fill available predictions from overlapping windows
        for j in range(i):
            row_idx = j
            step = i - j
            pred_col = f"{step}-pred" if f"{step}-pred" in predictions.columns else f"pred_{step}"
            record[f"pred_{step}"] = predictions.iloc[row_idx].get(pred_col, np.nan)

        # Add observed value if available
        record["real"] = y_real.get(target_date, np.nan)
        records.append(record)

    # --- Full horizon predictions for the middle of the series ---
    for i in range(len(predictions) - horizon_size + 1):
        record = {}
        target_date = predictions.index[i] + pd.Timedelta(days=horizon_size)
        record["date"] = target_date

        for j in range(horizon_size):
            row_idx = i + j
            step = horizon_size - j
            pred_col = f"{step}-pred" if f"{step}-pred" in predictions.columns else f"pred_{step}"
            record[f"pred_{step}"] = predictions.iloc[row_idx].get(pred_col, np.nan)

        record["real"] = y_real.get(target_date, np.nan)
        records.append(record)

    # --- Partial forecasts at the end of the series (less than full horizon) ---
    for i in range(horizon_size - 1, 0, -1):  # reversed order to maintain date sequence
        base_idx = len(predictions) - i
        cutoff_date = predictions.index[base_idx]
        target_date = cutoff_date + pd.Timedelta(days=horizon_size)
        record = {"date": target_date}

        # Initialize all steps as NaN
        for step in range(1, horizon_size + 1):
            record[f"pred_{step}"] = np.nan

        # Fill available predictions only
        for j in range(i):
            row_idx = base_idx + j
            if row_idx >= len(predictions):
                continue
            step = horizon_size - j
            pred_col = f"{step}-pred" if f"{step}-pred" in predictions.columns else f"pred_{step}"
            record[f"pred_{step}"] = predictions.iloc[row_idx].get(pred_col, np.nan)

        record["real"] = y_real.get(target_date, np.nan)
        records.append(record)

    # Assemble the reconstructed DataFrame and save to CSV
    df_records = pd.DataFrame(records)
    df_records.set_index("date", inplace=True)

    suffix = "with_covariates" if uses_covariates else "without_covariates"
    file_name = f"{model_name}_{train_size}_{input_size}_{horizon_size}_{suffix}.csv"
    file_path = os.path.join(predictions_dir, file_name)

    try:
        df_records.to_csv(
            file_path,
            index=True,
            float_format="%.2f",
            date_format="%Y-%m-%d"
        )
        print(f"Saved predictions to: {file_path}")
    except Exception as e:
        raise IOError(f"Failed to save predictions to CSV. Reason: {e}")