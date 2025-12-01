import pandas as pd
from pathlib import Path

FILE_NAME = "models_runtime.csv"

def save_time(
    model_name: str,
    start_year: int,
    end_year: int,
    input_size: int,
    train_size: int,
    horizon_size: int,
    offset_days: int,
    fit_time: float,
    predict_time: float,
    uses_covariates: bool,
    sort_by_runtime: bool,
    timing_dir: str, 
) -> pd.DataFrame:
    """
    Create or update a benchmarking CSV file recording model runtimes in the specified directory.

    If the CSV file does not exist, it is created with the current benchmark.
    If it exists, it updates the timing metrics for matching records or appends a new record.
    Optionally sorts the CSV and returned DataFrame by total runtime (fit_time + predict_time)
    from fastest to slowest.

    Parameters
    ----------
    model_name : str
        Name of the forecasting model.

    start_year : int
        Starting year of the evaluation period.

    end_year : int
        Ending year of the evaluation period.

    input_size : int
        Length of the input context window used in the model.

    train_size : int
        Number of training samples used.

    horizon_size : int
        Number of steps predicted by the model.
        
    offset_days: int
        Number of days added to or subtracted from a given date to shift it forward or backward in time.

    fit_time : float
        Time taken to fit/train the model (in seconds).

    predict_time : float
        Time taken to generate predictions (in seconds).

    uses_covariates : bool
        Whether the model used exogenous variables.

    sort_by_runtime : bool, optional, default=True
        If True, sort the CSV and returned DataFrame by total runtime ascending.
            
    timing_dir : str
        Directory path where the benchmarking CSV file will be stored.

    Returns
    -------
    pd.DataFrame
        The full benchmarking DataFrame after the update or creation,
        optionally sorted by total runtime (fit_time + predict_time).

    Raises
    ------
    IOError
        If there is an issue reading from or writing to the CSV file.
    """
    file_path = Path(timing_dir) / FILE_NAME

    new_row = pd.DataFrame(
        {
            "model_name": [model_name],
            "uses_covariates": [uses_covariates],
            "train_size": [train_size],
            "input_size": [input_size],
            "horizon_size": [horizon_size],
            "offset_days": [offset_days],
            "start_year": [start_year],
            "end_year": [end_year],
            "fit_time": [fit_time],
            "predict_time": [predict_time],
        }
    )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        if sort_by_runtime:
            sorted_new_row = new_row.sort_values(by=["fit_time", "predict_time"])
            sorted_new_row.to_csv(file_path, index=False)
            return sorted_new_row
        else:
            new_row.to_csv(file_path, index=False)
            return new_row

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Failed to read timing CSV file at {file_path}. Reason: {e}")

    mask = (
        (df["model_name"] == model_name)
        & (df["uses_covariates"] == uses_covariates)
        & (df["train_size"] == train_size)
        & (df["input_size"] == input_size)
        & (df["horizon_size"] == horizon_size)
        & (df["offset_days"] == offset_days)
        & (df["start_year"] == start_year)
        & (df["end_year"] == end_year)
    )

    if mask.any():
        idx = df.index[mask][0]
        df.at[idx, "fit_time"] = fit_time.round(2)
        df.at[idx, "predict_time"] = predict_time.round(2)
    else:
        df = pd.concat([df, new_row], ignore_index=True)

    if sort_by_runtime:
        df["total_time"] = df["fit_time"] + df["predict_time"]
        df = df.sort_values(by="total_time", ascending=True).drop(columns=["total_time"])

    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save timing CSV file at {file_path}. Reason: {e}")

    return df