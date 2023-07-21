import sys
from typing import Any, Callable, Dict, Optional, Tuple
from darts import TimeSeries, concatenate
from darts.metrics import rmse
import pandas as pd
import logging
from glob import glob
import pyspark
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import hyperopt
# from hyperopt.pyll.base import scope

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.xgb_utils.xgb_train import xgb_train_function
from tsboi.mlflow_models.ohlcv_models import MLflowXGBOHLCVModel

MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data/cleaned')
# FREQ = '1T'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_train_objective(
        series_dict: Dict[str, TimeSeries],
        covariates_dict: Dict[str, TimeSeries],
        lags_dict: Dict[str, int],
        probabilistic_dict: Dict[str, Any],
        mlflow_nested: bool = True,
        random_state: Optional[int] = None) \
        -> Callable:

    def train_function(
            params) \
            -> Dict[str, float]:

        with mlflow.start_run(nested=mlflow_nested) as run:

            mlflow.log_params(params=params)

            model, prediction = \
                xgb_train_function(
                    series_dict=series_dict,
                    covariates_dict=covariates_dict,
                    lags_dict=lags_dict,
                    probabilistic_dict=probabilistic_dict,
                    random_state=random_state,
                    **params
                )
            rmse_score = rmse(series_dict["series_test"], prediction)
            metrics = {"rmse_val": rmse_score}
            mlflow.log_metrics(metrics)

        return {'status': hyperopt.STATUS_OK, 'loss': metrics['rmse_val']}

    return train_function


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


def log_best(
        run,
        metric: str) \
        -> Tuple[str, float]:

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))

    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])

    return best_run.info.run_id, best_run.data.metrics[metric]




def main() \
        -> None:
    random_state = 42

    series, covariates = load_dataset()
    series_train_val, series_test = series.split_after(split_point=0.8)
    covariates_train_val, covariates_test = covariates.split_after(split_point=0.8)
    series_train, series_val = series_train_val.split_after(split_point=0.8)
    covariates_train, covariates_val = covariates_train_val.split_after(split_point=0.8)

    series_dict = \
        {
            'series_train': series_train,
            'series_val': series_val,
            'series_test': series_val
        }
    covariates_dict = \
        {
            'covariates': covariates,
            'covariates_train': covariates_train,
            'covariates_val': covariates_val
        }
    lags_dict = {"lags_past_covariates": 60}
    probabilistic_dict = {}

    space = {
        'learning_rate': hyperopt.hp.loguniform('learning_rate', -3, 0),
        'max_depth': hyperopt.hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9]),
        'reg_alpha': hyperopt.hp.loguniform('reg_alpha', -4, 0),
        'n_estimators': hyperopt.hp.randint('n_estimators', 100, 1000),
    }

    trials = hyperopt.SparkTrials(parallelism=1)
    train_objective = \
        build_train_objective(
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            lags_dict=lags_dict,
            probabilistic_dict=probabilistic_dict,
            mlflow_nested=True,
            random_state=random_state
        )

    with mlflow.start_run(run_name=f"{MODEL_NAME}-hp-tune") as run:
        hyperopt.fmin(
            fn=train_objective,
            space=space,
            algo=hyperopt.tpe.suggest,
            max_evals=3,
            trials=None
        )
        best_child_run_id, best_child_run_metrics = log_best(run, 'rmse_val')
        logger.info(f"Best run: {best_child_run_id} with metrics: {best_child_run_metrics}")

        # get best run
        client = mlflow.tracking.MlflowClient()
        best_child_run = client.get_run(best_child_run_id)
        best_child_run_params = best_child_run.data.params

        logger.info(f"Best run params: {best_child_run_params}")

    # refit
    with mlflow.start_run(run_name=f"{MODEL_NAME}-refit") as run:
        series_refit_dict = \
            {
                'series_train': series_train_val,
                'series_val': series_train_val,
                'series_test': series_test
            }
        covariates_refit_dict = \
            {
                'covariates': covariates,
                'covariates_train': covariates_train_val,
                'covariates_val': covariates_train_val
            }

        # TODO: better handle this
        # all values to float or int, if possible
        for key, value in best_child_run_params.items():
            try:
                best_child_run_params[key] = int(value)
            except ValueError:
                try:
                    best_child_run_params[key] = float(value)
                except ValueError:
                    pass

        mlflow.log_params(best_child_run_params)

        model, prediction = xgb_train_function(
            series_dict=series_refit_dict,
            covariates_dict=covariates_refit_dict,
            lags_dict=lags_dict,
            probabilistic_dict=probabilistic_dict,
            random_state=random_state,
            **best_child_run_params
        )

        print(f"Test RMSE: {rmse(series_test, prediction)}")

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

        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=MLflowXGBOHLCVModel(),
            artifacts=artifacts,
        )

        # clean up
        shutil.rmtree(MODEL_DIR)


if __name__ == '__main__':
    main()
