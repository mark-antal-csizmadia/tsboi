import sys
import logging
import mlflow
from typing import Dict, Optional
from pathlib import Path
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path('data/cleaned')

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_df():
    paths = sorted(glob(str(DATA_DIR / '*.csv')))
    # TODO: remove
    logger.info("Loading %s files from %s", len(paths), DATA_DIR)
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs, axis=0)
    logger.info("Loaded %s rows", df.shape[0])

    return df


if __name__ == "__main__":
    n_timesteps = 2000

    logged_model = 'runs:/b0373404121a4aeda53b57ddb6faa834/ohlcv-xgb-20230728150827'
    model_info = mlflow.artifacts.load_dict(f'{logged_model}/artifacts/model_info.json')
    target_id: str = model_info['target_id']
    covariate_ids: str = model_info['covariate_ids']
    extreme_lags: Dict[str, Optional[int]] = model_info['model.extreme_lags']
    num_samples: int = model_info['num_samples']
    logger.info("model_info: %s", model_info)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # TODO: too manual here
    df = get_df()
    df = df.iloc[70000:80000]
    # df = df.iloc[-6320:]
    logger.info("df start ts: %s", df.iloc[0]['ts'])
    logger.info("df end ts: %s", df.iloc[0]['ts'])

    min_past_lag = min(extreme_lags["min_past_covariate_lag"], extreme_lags["min_target_lag"]) \
        if extreme_lags["min_target_lag"] else extreme_lags["min_past_covariate_lag"]
    max_past_lag = abs(min_past_lag)
    logger.info("max_past_lag: %s", max_past_lag)

    predictions = []
    dfs = []

    tss = []
    for i in range(n_timesteps):
        from_idx = i
        to_idx = from_idx + max_past_lag
        logger.info(f"Predicting next 1 timestep at {df.iloc[to_idx]['ts']}, "
                    f"with covariates/target lags from {df.iloc[from_idx]['ts']} to {df.iloc[to_idx]['ts']}")

        prediciton = loaded_model.predict(df.iloc[from_idx:to_idx])

        dfs.append(df.iloc[from_idx:to_idx])
        predictions.append(prediciton)

    df_pred = pd.concat(predictions, axis=0)
    df_pred.set_index('ts', inplace=True)
    # outer join on ts as index
    df["ts"] = df["ts"].astype('datetime64[ns]')
    df.set_index('ts', inplace=True)
    df = df_pred.join(df, how='inner')
    print(df.head())

    # if close std is not 0, then calculate upper and lower bounds
    if df['close_std'].sum() > 0:
        df['close_plus_std'] = df_pred['close_mean'] + df_pred['close_std']
        df['close_minus_std'] = df_pred['close_mean'] - df_pred['close_std']
    df.drop(columns=["close_std", "open", "high", "low", "volume"], inplace=True)

    df.plot()
    plt.show()
