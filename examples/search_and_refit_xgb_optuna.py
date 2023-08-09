import sys
import logging
import shutil
from datetime import datetime
import argparse
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import rmse
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import optuna
from optuna.integration.mlflow import MLflowCallback


from tsboi.trainers.xgb_train import xgb_train_function
from tsboi.mlflow_models.darts_xgb import MLflowDartsXGBModel
from tsboi.data.base_dataset import BaseDataset


MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_PATH = MODEL_DIR / 'model.pkl'
MODEL_INFO_PATH = MODEL_DIR / 'model_info.json'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data/split')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for hyperparameter search.')
    parser.add_argument('--n_trials', help='Number of trials', type=int, default=10)
    parser.add_argument('--dataset_digest', help='Latest commit when data.dvc was updated', type=str, required=True)
    parser.add_argument('--random_state', help='Random state for reproducibility', type=int, default=42)

    return parser


def objective(
        trial: optuna.Trial,
        series_dict: Dict[str, TimeSeries],
        covariates_dict: Dict[str, TimeSeries],
        lags_dict: Dict[str, int],
        probabilistic_dict: Dict[str, Any],
        random_state: Optional[int] = None) \
        -> float:

    learning_rate = 1.2
    max_depth = trial.suggest_int('max_depth', 3, 9)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 1e-1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'reg_alpha': reg_alpha,
        'n_estimators': n_estimators,
    }

    model = \
        xgb_train_function(
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            lags_dict=lags_dict,
            probabilistic_dict=probabilistic_dict,
            random_state=random_state,
            **params
        )
    prediction = model.predict(
        n=len(series_dict["series_test"]),
        past_covariates=covariates_dict["covariates"],
        num_samples=probabilistic_dict.get("num_samples", 1),
    )

    return rmse(series_dict["series_test"], prediction)


def main():
    target_id = 'close'
    covariate_ids = ['open', 'high', 'low']

    dataset = BaseDataset(
        data_dir=DATA_DIR,
        file_pattern='*.csv',
        target_id=target_id,
        covariate_ids=covariate_ids,
        dtype='float32')

    series_train, covariates_train = dataset.load_dataset(subset='train')
    series_val, covariates_val = dataset.load_dataset(subset='val')
    series_test, covariates_test = \
        dataset.load_dataset(subset='test', record_description=True, record_examples_df_n_timesteps=100)

    series_train_val = series_train.concatenate(series_val, axis=0)
    covariates_train_val = covariates_train.concatenate(covariates_val, axis=0)
    series = series_train_val.concatenate(series_test, axis=0)
    covariates = covariates_train_val.concatenate(covariates_test, axis=0)

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
    lags_dict = {"lags_past_covariates": 10}
    probabilistic_dict = {}

    mlflc = MLflowCallback(
        tracking_uri='mlruns',
        metric_name="rmse_val",
    )

    study = optuna.create_study(study_name=MODEL_NAME, direction='minimize')
    study.optimize(
        lambda trial: objective(
            trial,
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            lags_dict=lags_dict,
            probabilistic_dict=probabilistic_dict,
            random_state=random_state
        ),
        n_trials=n_trials,
        callbacks=[mlflc]
    )

    best_trial = study.best_trial
    logger.info('Best trial: score {}, params {}'.format(best_trial.value, best_trial.params))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    n_trials = args.n_trials
    random_state = args.random_state
    dataset_digest = args.dataset_digest
    main()
