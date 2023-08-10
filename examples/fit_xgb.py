import shutil
import logging
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.data.pandas_dataset import PandasDataset
from mlflow import MlflowClient
from darts.metrics import rmse
from dotenv import load_dotenv, find_dotenv

from tsboi.trainers.xgb_train import xgb_train_function
from tsboi.mlflow_models.darts_xgb import MLflowDartsXGBModel
from tsboi.data.base_dataset import BaseDataset

MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'model.pkl'
MODEL_INFO_PATH = MODEL_DIR / 'model_info.json'
DATA_DIR = Path('data/split')

load_dotenv(find_dotenv())
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for cleaning OHLCV data for model training.')
    parser.add_argument('--dataset_digest', help='Latest commit when data.dvc was updated', type=str, required=True)
    parser.add_argument('--random_state', help='Random state for reproducibility', type=int, default=42)

    return parser


def main() \
        -> None:
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

    # TODO: remove ignore_time_axis=True
    series_train_val = series_train.concatenate(series_val, axis=0, ignore_time_axis=True)
    covariates_train_val = covariates_train.concatenate(covariates_val, axis=0, ignore_time_axis=True)
    series = series_train_val.concatenate(series_test, axis=0, ignore_time_axis=True)
    covariates = covariates_train_val.concatenate(covariates_test, axis=0, ignore_time_axis=True)

    mlflow_dataset: PandasDataset = \
        mlflow.data.from_pandas(df=dataset.examples_df, source='/tmp/dvcstore/', digest=args.dataset_digest)

    logger.info(f"Dataset description:")
    logger.info(f"{dataset.description}")

    run_params = \
        {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 3,#200,
            'reg_alpha': 0.1,
        }

    series_dict = \
        {
            'series_train': series_train_val,
            'series_val': series_train_val,
            'series_test': series_test
        }
    covariates_dict = \
        {
            'covariates': covariates,
            'covariates_train': covariates_train_val,
            'covariates_val': covariates_train_val
        }
    lags_dict = {"lags_past_covariates": 60}
    probabilistic_dict = {}

    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking uri: {MLFLOW_TRACKING_URI}")

    mlflow_experiment_id = client.create_experiment(MODEL_NAME)
    logger.info(f"MLflow experiment id: {mlflow_experiment_id}")

    client.set_experiment_tag(experiment_id=mlflow_experiment_id, key="dataset_digest", value=args.dataset_digest)

    experiment = client.get_experiment(mlflow_experiment_id)
    logger.info("Artifact Location: {}".format(experiment.artifact_location))

    run_object = client.create_run(experiment_id=mlflow_experiment_id, tags={"mlflow.runName": f"{MODEL_NAME}-fit"})

    with mlflow.start_run(run_id=run_object.info.run_id) as run:
        mlflow.log_input(dataset=mlflow_dataset, context="training")
        mlflow.log_params(run_params)

        model = xgb_train_function(
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            lags_dict=lags_dict,
            probabilistic_dict=probabilistic_dict,
            random_state=random_state,
            **run_params
        )

        prediction = model.predict(
            n=len(series_dict["series_test"]),
            past_covariates=covariates_dict["covariates"],
            num_samples=probabilistic_dict.get("num_samples", 1),
        )

        logger.info(f"Test RMSE: {rmse(series_test, prediction):4f}")

        # TODO: remove this
        series_train_val.plot(label='train_val')
        series_test.plot(label='test')
        prediction.plot(label='prediction')
        plt.legend()
        plt.show()
        # END TODO

        model.save(str(MODEL_PATH))
        model_info = {
            "target_id": target_id,
            "covariate_ids": covariate_ids,
            "model.extreme_lags": {
                "min_target_lag": model.extreme_lags[0], "max_target_lag": model.extreme_lags[1],
                "min_past_covariate_lag": model.extreme_lags[2], "max_past_covariate_lag": model.extreme_lags[3],
                "min_future_covariate_lag": model.extreme_lags[4], "max_future_covariate_lag": model.extreme_lags[5]
            },
            "num_samples": probabilistic_dict.get("num_samples", 1),
        }
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f)

        artifacts = {
            "path_to_model_file": str(MODEL_PATH),
            "path_to_model_info_file": str(MODEL_INFO_PATH)
        }
        input_schema = Schema(
            [
                ColSpec("datetime", "ts"),
                ColSpec("double", target_id)
            ] +
            [
                ColSpec("double", covariate_id) for covariate_id in covariate_ids
            ]
        )
        output_schema = Schema(
            [
                ColSpec("datetime", "ts"),
                ColSpec("double", f"{target_id}_mean"),
                ColSpec("double", f"{target_id}_std"),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = dataset.examples_df.iloc[:max(lags_dict.values())]

        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=MLflowDartsXGBModel(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            code_path=["tsboi"],
            pip_requirements="requirements.txt"
        )

        # TODO: uncomment this
        # shutil.rmtree(MODEL_DIR)


if __name__ == '__main__':
    args = get_parser().parse_args()
    random_state = args.random_state
    main()
