import shutil
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.data.pandas_dataset import PandasDataset
from darts.metrics import rmse
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from tsboi.trainers.transformer_train import transformer_train_function
from tsboi.mlflow_models.darts_transformer import MLflowDartsTransformerModel
from tsboi.data.base_dataset import BaseDataset

MODEL_NAME = 'ohlcv-transformer-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TARGET_PIPELINE_PATH = MODEL_DIR / 'pipeline_target.pkl'
PAST_COVARIATES_PIPELINE_PATH = MODEL_DIR / 'past_covariates_pipeline.pkl'
MODEL_INFO_PATH = MODEL_DIR / 'model_info.json'
DATA_DIR = Path('data/split')

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

    mlflow_dataset: PandasDataset = \
        mlflow.data.from_pandas(df=dataset.examples_df, source='/tmp/dvcstore/', digest=args.dataset_digest)

    logger.info(f"Dataset description:")
    logger.info(f"{dataset.description}")

    pipeline_target = Pipeline([MissingValuesFiller(), Scaler(RobustScaler())])
    series_train_val_preprocessed = pipeline_target.fit_transform(series_train_val)
    series_test_preprocessed = pipeline_target.transform(series_test)
    joblib.dump(pipeline_target, TARGET_PIPELINE_PATH)

    pipeline_past_covariates = Pipeline([MissingValuesFiller(), Scaler(MinMaxScaler())])
    covariates_train_val_preprocessed = pipeline_past_covariates.fit_transform(covariates_train_val)
    covariates_test_preprocessed = pipeline_past_covariates.transform(covariates_test)
    covariates_preprocessed = pipeline_past_covariates.transform(covariates)
    joblib.dump(pipeline_past_covariates, PAST_COVARIATES_PIPELINE_PATH)

    run_params = \
        {
            'n_epochs': 2,
        }

    series_dict = \
        {
            'series_train': series_train_val_preprocessed,
            'series_val': series_train_val_preprocessed,
            'series_test': series_test_preprocessed
        }
    covariates_dict = \
        {
            'covariates': covariates_preprocessed,
            'covariates_train': covariates_train_val_preprocessed,
            'covariates_val': covariates_train_val_preprocessed
        }
    lags_dict = {"lags": 60}

    with mlflow.start_run(run_name=f"{MODEL_NAME}-fit") as run:
        mlflow.log_input(dataset=mlflow_dataset, context="training")
        mlflow.log_params(run_params)

        model = transformer_train_function(
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            lags_dict=lags_dict,
            random_state=random_state,
            model_name=MODEL_NAME,
            **run_params
        )

        prediction = model.predict(
            n=1000,
            past_covariates=covariates_dict["covariates"],
        )

        prediction = pipeline_target.inverse_transform(prediction, partial=True)

        print(f"Test RMSE: {rmse(series_test, prediction)}")

        # TODO: remove this
        series_train_val.plot(label='train_val')
        series_test.plot(label='test')
        prediction.plot(label='prediction')
        plt.legend()
        plt.show()
        # END TODO

        model_info = {
            "target_id": target_id,
            "covariate_ids": covariate_ids,
            "model.extreme_lags": {
                "min_target_lag": model.extreme_lags[0], "max_target_lag": model.extreme_lags[1],
                "min_past_covariate_lag": model.extreme_lags[2], "max_past_covariate_lag": model.extreme_lags[3],
                "min_future_covariate_lag": model.extreme_lags[4], "max_future_covariate_lag": model.extreme_lags[5]
            },
            "num_samples": 1,
        }
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f)

        artifacts = {
            "path_to_model_file": str(Path("darts_logs") / MODEL_NAME),
            "path_to_model_info_file": str(MODEL_INFO_PATH),
            "path_to_pipeline_target_file": str(TARGET_PIPELINE_PATH),
            "path_to_pipeline_past_covariates_file": str(PAST_COVARIATES_PIPELINE_PATH),
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
            python_model=MLflowDartsTransformerModel(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            code_path=["tsboi"],
            pip_requirements="requirements.txt"
        )

        shutil.rmtree(MODEL_DIR)


if __name__ == '__main__':
    args = get_parser().parse_args()
    random_state = args.random_state
    main()
