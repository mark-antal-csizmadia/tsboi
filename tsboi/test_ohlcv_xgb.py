import sys
import logging
import mlflow
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
    logged_model = 'runs:/15be9ddd93db4134a330b2218e0975e4/ohlcv-xgb-20230725163548'
    n_timesteps = 200

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    df = get_df()
    df = df.iloc[70000:80000]

    predictions = []
    dfs = []

    lag = 60
    tss = []
    for i in range(n_timesteps):
        from_idx = i
        to_idx = from_idx + lag

        prediciton = loaded_model.predict(df.iloc[from_idx:to_idx])
        dfs.append(df.iloc[from_idx:to_idx])
        predictions.append(prediciton)

    df_pred = pd.concat(predictions, axis=0)
    df_pred.rename(
        columns={"prediction_mean": 'close_pred', 'prediction_std': 'close_pred_std', 'prediction_timestamp': 'ts'},
        inplace=True)
    df_pred.set_index('ts', inplace=True)

    df_pred['close_pred_top'] = df_pred['close_pred'] + df_pred['close_pred_std']
    df_pred['close_pred_bottom'] = df_pred['close_pred'] - df_pred['close_pred_std']

    # outer join on ts as index
    df["ts"] = df["ts"].astype('datetime64[ns]')
    df.set_index('ts', inplace=True)
    df = df_pred.join(df, how='inner')
    print(df.head())

    df.drop(columns=["open", "high", "low", "volume", "close_pred_std"], inplace=True)
    df.plot()
    plt.show()
