import pandas as pd
import numpy as np
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection import SlidingWindowSplitter
from save_predictions import save_predictions
from save_time import save_time
import time
from datetime import date

def predict(
    model: BaseForecaster,
    model_name: str,
    start_year: int,
    end_year: int,
    y: pd.Series,
    X: pd.DataFrame,
    input_size: int,
    train_size: int,
    offset_days: int,
    splitter: SlidingWindowSplitter,
    start_date: date,
    amount_missing_data: int,
    predictions_dir: str,
    timing_dir: str
) -> tuple[pd.DataFrame, float, float]:
    """
    Predict a time series forecasting model using a sliding window cross-validation strategy.

    The prediction process:
    - Optionally pre-trains the model with a static window of `train_size`.
    - Iteratively predicts using the defined sliding window strategy.
    - Supports the use of exogenous variables (`X`) if the model accepts them.
    - Optionally continues predictions after the end of the time series (`amount_missing_data`).
    - Saves predictions and timing information to specified directories.

    Parameters
    ----------
    model : BaseForecaster
        A forecaster compatible with the sktime interface.

    model_name : str
        Identifier for saving outputs.

    start_year : int
        Start year for logging and metadata.

    end_year : int
        End year for logging and metadata.

    y : pd.Series
        Univariate time series with datetime-like index.

    X : pd.DataFrame
        Exogenous variables aligned with `y`, or None.

    input_size : int
        Number of past observations to use as input.

    train_size : int
        Number of initial observations to train on (static fit); 0 means no initial training.

    splitter : SlidingWindowSplitter
        Sliding window splitter defining train/test splits.

    start_date : date
        Date from which prediction starts (used to skip earlier folds).

    amount_missing_data : int
        Number of forecast steps to add at the end if the last window doesn’t cover all required horizons.

    predictions_dir : str
        Directory path to save forecast results.

    timing_dir : str
        Directory path to save timing information.

    Returns
    -------
    tuple
        Tuple containing:
        - pd.DataFrame: Predictions in wide format with forecast horizon columns.
        - float: Total training time.
        - float: Total prediction time.

    Raises
    ------
    TypeError
        If `model` is not a BaseForecaster instance.
        If `y` is not a pandas Series.
        If `splitter` is not a None.
        If `start_date` is not a `date` object.
        If `X` is provided and is not a DataFrame.
    
    ValueError
        If `model_name` is not a non-empty string.
        If `start_year` or `end_year` are not integers.
        If `start_year` is greater than `end_year`.
        If `y` is empty.
        If `X` is provided and its length doesn't match `y`.
        If `input_size` is not a positive integer.
        If `train_size` is negative.
        If `amount_missing_data` is not an integer.
        If `predictions_dir` or `timing_dir` are not valid string paths.
    """

    # ------------------- Input Validation ------------------- #
    if model is None or not isinstance(model, BaseForecaster):
        raise TypeError("`model` must be an instance of sktime BaseForecaster.")
    
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("`model_name` must be a non-empty string.")
    
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("`start_year` and `end_year` must be integers.")
    
    if start_year > end_year:
        raise ValueError("`start_year` must be <= `end_year`.")
    
    if y is None or not isinstance(y, pd.Series) or y.empty:
        raise TypeError("`y` must be a non-empty pandas Series.")
    
    if X is not None and len(X) != len(y):
        raise ValueError("If provided, `X` must match length of `y`.")
    
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("`input_size` must be a positive integer.")

    if not isinstance(train_size, int) or train_size < 0:
        raise ValueError("`train_size` must be a non-negative integer.")
    
    if not splitter is not None:
        raise TypeError("`splitter` must be a SlidingWindowSplitter, not None.")

    if not isinstance(start_date, date):
        raise TypeError("`start_date` must be a date object.")

    if not isinstance(amount_missing_data, int):
        raise TypeError("`amount_missing_data` must be an integer.")

    if not isinstance(predictions_dir, str):
        raise TypeError("`predictions_dir` must be a string path.")

    if not isinstance(timing_dir, str):
        raise TypeError("`timing_dir` must be a string path.")

    # ------------------- Setup ------------------- #
    y = y.copy()
    if X is not None:
        X = X.copy()

    fh = splitter.get_fh()
    results = []
    cutoff_dates = []
    start_date = pd.Timestamp(start_date)

    # ------------------- Optional Static Training ------------------- #
    if train_size > 0:
        train_idx = list(range(0, y.index.get_loc(start_date)))
        y_train = y.iloc[train_idx]
        cutoff = y.index[train_idx[-1]]

        print("Training:", y_train.index[0], "-", y_train.index[-1])

        if X is not None:
            X_train = X.iloc[train_idx]
            start_fit = time.perf_counter()
            model.fit(y=y_train, X=X_train, fh=fh)
            fit_time = time.perf_counter() - start_fit

        else:
            start_fit = time.perf_counter()
            model.fit(y=y_train, fh=fh)
            fit_time = time.perf_counter() - start_fit

    else:
        # Cold start: initialize model with empty fit (e.g., for online updates)
        empty_y = y.iloc[:0]
        empty_X = X.iloc[:0] if X is not None else None

        start_fit = time.perf_counter()
        model.fit(y=empty_y, X=empty_X, fh=fh)
        fit_time = time.perf_counter() - start_fit

    # ------------------- Rolling Forecast Prediction ------------------- #
    predict_time = 0
    print("Predicting:", start_date, "-", y.index[-1])
    start_loc = y.index.get_loc(start_date)

    for train_idx, test_idx in splitter.split(y):
        if train_idx[0] < start_loc:
            continue

        y_train = y.iloc[train_idx]
        cutoff = y.index[train_idx[-1]]

        if X is not None:
            X_sub = X.iloc[list(train_idx) + list(test_idx)]
            start_predict = time.perf_counter()
            y_pred = model.predict(y=y_train, X=X_sub)
            predict_time += time.perf_counter() - start_predict

        else:
            start_predict = time.perf_counter()
            y_pred = model.predict(y=y_train)
            predict_time += time.perf_counter() - start_predict

        # Ensure Series output
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred.ravel(), index=list(fh.to_absolute(y.index)))

        cutoff_dates.append(cutoff)
        results.append(y_pred.values)

    # ------------------- Final Forecasts for Incomplete Horizon ------------------- #
    if train_idx is not None and test_idx is not None and amount_missing_data != 0:
        for i in range(1, len(fh)):
            shifted_train_idx = range(train_idx[0] + i, train_idx[-1] + i + 1)

            if max(shifted_train_idx) >= len(y):
                break

            y_train = y.iloc[shifted_train_idx]

            start_predict = time.perf_counter()
            y_pred = model.predict(y=y_train)
            predict_time += time.perf_counter() - start_predict

            if isinstance(y_pred, np.ndarray):
                y_pred = pd.Series(y_pred.ravel(), index=list(fh.to_absolute(y.index)))

            cutoff = y.index[shifted_train_idx[-1]]
            cutoff_dates.append(cutoff)
            results.append(y_pred.values)

    # ------------------- Format Predictions ------------------- #
    predictions = pd.DataFrame(results, index=cutoff_dates)
    predictions.columns = [f"{i + 1}-pred" for i in range(predictions.shape[1])]
    predictions.index.name = "cutoff"
    predictions[predictions < 0] = 0  # Enforce non-negativity

    # ------------------- Save Outputs ------------------- #
    uses_covariates = not model.get_tag('ignores-exogeneous-X') if X is not None else False

    save_predictions(
        model_name=model_name,
        predictions=predictions,
        y_real=y,
        train_size=train_size,
        input_size=input_size,
        horizon_size=len(fh),
        uses_covariates=uses_covariates,
        predictions_dir=predictions_dir,
    )

    save_time(
        model_name=model_name,
        start_year=start_year,
        end_year=end_year,
        input_size=input_size,
        train_size=train_size,
        horizon_size=len(fh),
        offset_days = offset_days,
        fit_time=fit_time,
        predict_time=predict_time,
        uses_covariates=uses_covariates,
        timing_dir=timing_dir,
        sort_by_runtime = False
    )

    return predictions, fit_time, predict_time
