import os
import re
import pandas as pd
from get_mps import get_mps_by_percentage, get_mps_by_threshold
from non_leap_date_utils import calculate_non_leap_offset_dates
from typing import Optional

def evaluate(
    start_year: int,
    end_year: int,
    en: pd.Series,
    input_size: int,
    horizon_size: int,
    offset_days: int,
    apply_percentage: bool,
    predictions_dir: str,
    evaluations_dir: str,
    start_pct: Optional[float] = None,
    end_pct:  Optional[float] = None,
    threshold:  Optional[float] = None,
    consecutive_days:  Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Evaluate forecasting of Main Pollen Season (MPS) start/end dates over multiple years.

    Compares predicted MPS against observed pollen data using one of two methods:
    - Cumulative percent thresholds (`apply_percentage=True`), or
    - Absolute threshold with consecutive days (`apply_percentage=False`).

    Generates a CSV summary with per-model, per-year, per-step accuracy metrics.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive range of years to evaluate. Must satisfy start_year <= end_year.

    en : pd.Series
        Observed pollen counts, indexed by datetime. Must be non-empty.

    input_size : int
        Length of input window used by the models (e.g., number of past days used for prediction).

    horizon_size : int
        Number of days ahead predicted by the model (forecast horizon length).

    offset_days : int
        Number of days to shift the calendar to align predictions with true values.

    apply_percentage : bool
        Indicates which MPS detection method to apply:
        - True: use `get_mps_by_percentage`.
        - False: use `get_mps_by_threshold`.

    predictions_dir : str
        Path to directory containing model forecast CSV files.

    evaluations_dir : str
        Path to directory where the evaluation result CSV will be saved.

    start_pct, end_pct : float, optional
        Percent thresholds (between 0 and 1) for determining MPS start and end via the percentage method.
        Required if `apply_percentage=True`.

    threshold : float, optional
        Minimum pollen value used to define a threshold crossing.
        Required if `apply_percentage=False`.

    consecutive_days : int, optional
        Number of consecutive days above the threshold required to define MPS.
        Required if `apply_percentage=False`.

    Returns
    -------
        pd.DataFrame
        A DataFrame containing evaluation results per model, year, and prediction step,
        including true and estimated MPS boundaries and related accuracy metrics.

        Additionally, the results are saved as a CSV file in `evaluations_dir`.
        
    Raises
    ------
    ValueError
        If any of the following conditions are met:
        - `start_year` or `end_year` are not integers or start_year > end_year.
        - `en` is not a non-empty pandas Series.
        - `apply_percentage` is not a boolean.
        - `input_size` or `horizon_size` are not positive integers.
        - `offset_days` is not an integer.
        - `predictions_dir` does not exist or is not a directory.
        - If `apply_percentage=True` and either `start_pct` or `end_pct` is None.
        - If `apply_percentage=False` and either `threshold` or `consecutive_days` is None.
        - If no matching prediction files are found for the given configuration.
        - If any expected prediction column (e.g., 'pred_1') is missing from a forecast file.
    """


    # --- Validate inputs ---
    if not (isinstance(start_year, int) and isinstance(end_year, int)) or start_year > end_year:
        raise ValueError("start_year/end_year must be ints with start_year <= end_year.")
    if not isinstance(en, pd.Series) or en.empty:
        raise ValueError("'en' must be a non-empty pandas Series.")
    if not isinstance(apply_percentage, bool):
        raise ValueError("'apply_percentage' must be boolean.")
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("'input_size' must be a positive integer.")
    if not isinstance(horizon_size, int) or horizon_size <= 0:
        raise ValueError("'horizon_size' must be a positive integer.")
    if not isinstance(offset_days, int):
        raise ValueError("'offset_days' must be an integer.")
    if not os.path.isdir(predictions_dir):
        raise ValueError(f"Predictions directory '{predictions_dir}' does not exist.")
    os.makedirs(evaluations_dir, exist_ok=True)
    if apply_percentage:
        if start_pct is None or end_pct is None:
            raise ValueError("start_pct and end_pct required if apply_percentage=True.")
    else:
        if threshold is None or consecutive_days is None:
            raise ValueError("threshold and consecutive_days required if apply_percentage=False.")

    rows = []

    # --- Gather prediction files for given configuration ---
    pattern = re.compile(rf".*_{input_size}_{horizon_size}_.*\.csv$")
    files = [
        os.path.join(predictions_dir, f)
        for f in os.listdir(predictions_dir)
        if os.path.isfile(os.path.join(predictions_dir, f)) and pattern.match(f)
    ]
    if not files:
        raise ValueError(f"No forecast files for input_size={input_size}, horizon_size={horizon_size}")

    # --- Evaluate each year and each forecast file ---
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        for file in files:
            # Determine valid evaluation window
            sd, ed = calculate_non_leap_offset_dates(
                start_year=year, end_year=year,
                input_size=0, train_size=0,
                offset_days=offset_days, horizon_size=0
            )
            start_date, end_date = pd.Timestamp(sd), pd.Timestamp(ed)

            df = pd.read_csv(file, parse_dates=["date"]).set_index("date")
            df_clean = df.dropna()
            if df_clean.empty:
                print(f"⚠️ Empty predictions in {file}, skip.")
                continue

            # Adjust date bounds for available data
            first_pred, last_pred = df_clean.index[0], df_clean.index[-1]
            if first_pred > start_date:
                print(f"⚠️ Adjusting start from {start_date} to {first_pred}")
                start_date = first_pred
            if last_pred < end_date:
                print(f"⚠️ Adjusting end from {end_date} to {last_pred}")
                end_date = last_pred

            real = en.loc[start_date:end_date]
            if real.empty:
                print(f"⚠️ No observed data for year {year}, skip.")
                continue

            # Compute ground-truth MPS bounds
            if apply_percentage:
                real_start, real_end = get_mps_by_percentage(real, start_pct, end_pct)
            else:
                real_start, real_end = get_mps_by_threshold(real, threshold, consecutive_days)
            if real_start is None or real_end is None:
                print(f"⚠️ Could not compute MPS for year {year}, skip.")
                continue

            # Evaluate predictions by forecast step
            for step in range(1, horizon_size + 1):
                col = f"pred_{step}"
                if col not in df_clean.columns:
                    raise ValueError(f"Missing column '{col}' in {file}")
                pred_series = df_clean.loc[start_date:end_date, col]
                if pred_series.empty:
                    print(f"⚠️ No data in {file} for step {step}, skip.")
                    continue

                # Compute predicted MPS
                if apply_percentage:
                    est_start, est_end = get_mps_by_percentage(pred_series, start_pct, end_pct)
                else:
                    est_start, est_end = get_mps_by_threshold(pred_series, threshold, consecutive_days)
                if est_start is None or est_end is None:
                    print(f"⚠️ Incomplete predicted MPS in {file}, step {step}, skip.")

                # Parse metadata from filename
                fname = os.path.basename(file)
                parts = fname.split("_")
                model_name = parts[0]
                train_size = int(parts[1])
                uses_cov = parts[4] == "with"

                # Append metrics row
                rows.append({
                    "year": year,
                    "model_name": model_name,
                    "uses_covariates": uses_cov,
                    "train_size": train_size,
                    "pred_n": step,
                    "start": real_start,
                    "end_date": real_end,
                    "est_start": est_start,
                    "est_end": est_end,
                    "start_dev": abs((real_start - est_start).days) if est_start else None,
                    "end_dev": abs((real_end - est_end).days) if est_end else None,
                    "duration": abs((real_end - real_start).days),
                    "est_duration": abs((est_end - est_start).days) if est_start and est_end else None,
                })

    # --- Summarize and export ---
    if not rows:
        print("⚠️ No evaluation records generated.")
        return None

    df_out = pd.DataFrame(rows)
    df_out["total_dev"] = df_out[["start_dev", "end_dev"]].sum(axis=1)
    df_out["duration_dev"] = df_out["duration"] - df_out["est_duration"].fillna(0)

    suffix = "percentage" if apply_percentage else "threshold"        
    if apply_percentage:
        file_path = os.path.join( evaluations_dir,
            f"mps_evaluation_{suffix}_{start_year}-{end_year}_{offset_days}_{input_size}_{horizon_size}_{start_pct}_{end_pct}.csv")
    else:
        file_path = os.path.join( evaluations_dir,
            f"mps_evaluation_{suffix}_{start_year}-{end_year}_{offset_days}_{input_size}_{horizon_size}_{threshold}_{consecutive_days}.csv")

    df_out.to_csv(file_path, index=False, date_format="%Y-%m-%d")
    print(f"✅ Evaluation complete. Saved to: {file_path}")
    
    return df_out