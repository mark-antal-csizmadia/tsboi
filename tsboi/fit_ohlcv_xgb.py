import sys
from typing import Tuple
from darts import TimeSeries, concatenate
from darts.metrics import rmse
import pandas as pd
import logging
from glob import glob
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.xgb_utils.xgb_train import xgb_train_function
from tsboi.mlflow_models.ohlcv_models import MLflowXGBOHLCVModel, MLflowXGBOHLCVModelSignature
from tsboi.data.base_dataset import BaseDataset

MODEL_NAME = 'ohlcv-xgb-{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
MODEL_DIR = Path('models') / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data/cleaned')
# FREQ = '1T'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() \
        -> None:
    random_state = 42

    dataset = BaseDataset(
        data_dir=DATA_DIR,
        file_pattern='*.csv',
        target_id='close',
        covariate_ids=['open', 'high', 'low'],
        dtype='float32')

    series, covariates = dataset.load_dataset(limit=1000)

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
            signature=MLflowXGBOHLCVModelSignature.signature,
            input_example=MLflowXGBOHLCVModelSignature.input_example
        )

        # clean up
        shutil.rmtree(MODEL_DIR)


if __name__ == '__main__':
    main()
