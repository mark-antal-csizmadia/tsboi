import sys
import shutil
import logging
import yaml
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from darts.metrics import rmse

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.xgb_utils.xgb_train import xgb_train_function
from tsboi.mlflow_models.darts_xgb import MLflowDartsXGBModel
from tsboi.data.base_dataset import BaseDataset

MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'model.pkl'
MODEL_INFO_PATH = MODEL_DIR / 'model_info.yaml'
DATA_DIR = Path('data/cleaned')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() \
        -> None:
    random_state = 42
    target_id = 'close'
    covariate_ids = ['open', 'high', 'low']

    dataset = BaseDataset(
        data_dir=DATA_DIR,
        file_pattern='*.csv',
        target_id=target_id,
        covariate_ids=covariate_ids,
        dtype='float32')

    # series, covariates = dataset.load_dataset(limit=1510000, record_examples_df_n_timesteps=1000)
    series, covariates = dataset.load_dataset(limit=60000, record_examples_df_n_timesteps=1000)

    logger.info(f"Dataset description:")
    logger.info(f"{dataset.description}")

    series_train_val, series_test = series.split_after(split_point=0.8)
    covariates_train_val, covariates_test = covariates.split_after(split_point=0.8)
    series_train, series_val = series_train_val.split_after(split_point=0.8)
    covariates_train, covariates_val = covariates_train_val.split_after(split_point=0.8)

    run_params = \
        {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 1000,
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

    with mlflow.start_run(run_name=f"{MODEL_NAME}-fit") as run:

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

        print(f"Test RMSE: {rmse(series_test, prediction)}")

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
            "num_samples": probabilistic_dict.get("num_samples", 1),
        }
        with open(MODEL_INFO_PATH, 'w') as f:
            yaml.dump(model_info, f)

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
                ColSpec("datetime", "prediction_timestamp"),
                ColSpec("double", "prediction_mean"),
                ColSpec("double", "prediction_std"),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = dataset.examples_df.iloc[:max(lags_dict.values())]

        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=MLflowDartsXGBModel(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example
        )

        shutil.rmtree(MODEL_DIR)


if __name__ == '__main__':
    main()
