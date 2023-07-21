import sys
import shutil
import logging
from typing import Tuple, Dict, Any
from pathlib import Path
from glob import glob
from datetime import datetime
from darts import TimeSeries
from darts.models import XGBModel
from darts import concatenate
from darts.metrics import mape, smape, rmse
import pandas as pd
import mlflow
import optuna
import matplotlib.pyplot as plt


# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.mlflow_models.ohlcv_models import MLflowXGBOHLCVModel

MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data/cleaned')
# FREQ = '1T'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset() \
        -> Tuple[TimeSeries, TimeSeries]:

    paths = sorted(glob(str(DATA_DIR / '*.csv')))
    # TODO: remove
    paths = paths[0:1000]
    logger.info(f"Loading %s files from %s", len(paths), DATA_DIR)

    dfs = [pd.read_csv(path, parse_dates=['ts']) for path in paths]
    logger.info(f"Loaded %s files", len(dfs))

    df = pd.concat(dfs, axis=0)
    # remove timezone info
    df['ts'] = df['ts'].dt.tz_localize(None)
    df.set_index('ts', inplace=True)

    # open, high, low, close, volume to float32
    df['open'] = df['open'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')
    df['volume'] = df['volume'].astype('float32')

    # infer frequency from model_input
    freq = pd.infer_freq(df.index)
    assert freq is not None, "Could not infer frequency from model_input"
    logger.info(f"Inferred frequency: %s", freq)

    print(df.head())
    print(df.info())
    print(df.describe())

    series = TimeSeries.from_dataframe(df, value_cols="close", freq=freq)
    covariate_open = TimeSeries.from_dataframe(df, value_cols="open", freq=freq)
    covariate_high = TimeSeries.from_dataframe(df, value_cols="high", freq=freq)
    covariate_low = TimeSeries.from_dataframe(df, value_cols="low", freq=freq)
    # covariate_volume = TimeSeries.from_dataframe(covariates, value_cols="volume", freq=FREQ)

    covariates = concatenate([covariate_open, covariate_high, covariate_low], axis="component")

    logger.info(f"series.start_time: {series.start_time}")
    logger.info(f"series.end_time: {series.end_time}")
    logger.info(f"series.freq: {series.freq}")
    logger.info(f"series.duration: {series.duration}")

    logger.info(f"covariates.start_time: {covariates.start_time}")
    logger.info(f"covariates.end_time: {covariates.end_time}")
    logger.info(f"covariates.freq: {covariates.freq}")
    logger.info(f"covariates.duration: {covariates.duration}")

    return series, covariates


def objective(
        trial: optuna.Trial) \
        -> float:

    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-1, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    }
    model = XGBModel(
        lags=None,
        lags_past_covariates=60,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10,
        **params
    )

    model.fit(
        series=series_train,
        past_covariates=covariates_train,
        val_series=series_val,
        val_past_covariates=covariates_val,
        verbose=True)

    prediction = model.predict(
        n=len(series_val),
        past_covariates=covariates)

    return rmse(series_val, prediction)


def print_callback(
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial) \
        -> None:

    logger.info("Current value: %s, Current params: %s", trial.value, trial.params)
    logger.info("Best value: %s, Best params: %s", study.best_value, study.best_trial.params)


if __name__ == "__main__":
    series, covariates = load_dataset()

    series_train_val, series_test = series.split_after(split_point=0.8)
    covariates_train_val, covariates_test = covariates.split_after(split_point=0.8)
    series_train, series_val = series_train_val.split_after(split_point=0.8)
    covariates_train, covariates_val = covariates_train_val.split_after(split_point=0.8)

    # optimize hyperparameters by minimizing the sMAPE on the validation set
    study = optuna.create_study(direction="minimize")
    study.optimize(func=objective, n_trials=2, callbacks=[print_callback])

    # get the parameters for the best model
    params = study.best_trial.params
    print(params)

    # retrain the model with best parameters
    model = XGBModel(
        lags=None,
        lags_past_covariates=60,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10,
        **params
    )

    model.fit(
        series=series_train_val,
        past_covariates=covariates_train_val,
        val_series=series_train_val,
        val_past_covariates=covariates_train_val,
        verbose=True)

    # predict
    prediction = model.predict(
        n=len(series_test),
        past_covariates=covariates,
        num_samples=1
    )

    # eval
    print(f"MAPE: {mape(series_test, prediction)}")
    print(f"SMAPE: {smape(series_test, prediction)}")
    print(f"RMSE: {rmse(series_test, prediction)}")

    # plot
    series_train_val.plot(label='train_val')
    series_test.plot(label='test')
    prediction.plot(label='prediction')
    plt.legend()
    plt.show()

    # save the model
    model.save(str(MODEL_DIR / 'model.pkl'))

    artifacts = {
        "path_to_model_file": str(MODEL_DIR / 'model.pkl'),
    }

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=MLflowXGBOHLCVModel(),
            artifacts=artifacts,
        )

    # clean up
    shutil.rmtree(MODEL_DIR)
