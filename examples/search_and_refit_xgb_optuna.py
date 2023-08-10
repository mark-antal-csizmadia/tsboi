import os
import logging
import argparse
from typing import Any, Dict, Optional
from pathlib import Path
import joblib
import pickle
from darts import TimeSeries
from darts.metrics import rmse
from mlflow import MlflowClient
import optuna
from optuna.integration.mlflow import MLflowCallback
from dotenv import load_dotenv, find_dotenv


from tsboi.trainers.xgb_train import xgb_train_function
from tsboi.data.base_dataset import BaseDataset

DATA_DIR = Path('data/split')
OPTUNA_DIR = Path('optuna_studies')
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(find_dotenv())
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATASET_DIGEST = os.getenv("DATASET_DIGEST")
DATASET_DIGEST = DATASET_DIGEST[:8]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for hyperparameter search.')
    parser.add_argument('--study_name', help='Name of the study', type=str, required=True)
    parser.add_argument('--n_trials', help='Number of trials', type=int, default=10)
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

    learning_rate = trial.suggest_float('learning_rate', 0.1, 0.2, log=True)
    max_depth = trial.suggest_int('max_depth', 4, 8)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-2, 1e-1, log=True)
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
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

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

    # mlflow_dataset = mlflow.data.from_pandas(df=dataset.examples_df, source='/tmp/dvcstore/', digest=DATASET_DIGEST)

    logger.info(f"Dataset description:")
    logger.info(f"{dataset.description}")

    if OPTUNA_DIR / f"{study_name}.pkl" in OPTUNA_DIR.iterdir():
        study = joblib.load(OPTUNA_DIR / f"{study_name}.pkl")
        with open(OPTUNA_DIR / f"{study_name}_sampler.pkl", "rb") as fin:
            study.sampler = pickle.load(fin)
        logger.info('Loaded study {}'.format(study_name))
        logger.info('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
        mlflow_experiment = client.get_experiment_by_name(name=study_name)
        mlflow_experiment_id = mlflow_experiment.experiment_id
        logger.info(f"MLflow experiment id: {mlflow_experiment_id}")
    else:
        study = optuna.create_study(study_name=study_name, direction='minimize')
        logger.info('Created study {}'.format(study_name))

        mlflow_experiment_id = client.create_experiment(name=study_name)
        logger.info(f"MLflow experiment id: {mlflow_experiment_id}")
        client.set_experiment_tag(experiment_id=mlflow_experiment_id, key="dataset_digest", value=DATASET_DIGEST)

        experiment = client.get_experiment(mlflow_experiment_id)
        logger.info("Artifact Location: {}".format(experiment.artifact_location))

    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking uri: {MLFLOW_TRACKING_URI}")


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
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="rmse_val",
        mlflow_kwargs={
            "experiment_id": mlflow_experiment_id,
        }
    )

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

    joblib.dump(study, OPTUNA_DIR / f"{study_name}.pkl")
    with open(OPTUNA_DIR / f"{study_name}_sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    study_name = args.study_name
    n_trials = args.n_trials
    random_state = args.random_state
    main()
