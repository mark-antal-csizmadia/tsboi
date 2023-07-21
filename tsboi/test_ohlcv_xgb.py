import sys
import mlflow
from pathlib import Path
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path('data/cleaned')

FREQ = '1T'
# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO


def get_df():
    paths = sorted(glob(str(DATA_DIR / '*.csv')))
    # TODO: remove
    paths = paths[1000:2000]
    print(f"Loading {len(paths)} files from {DATA_DIR}")
    dfs = [pd.read_csv(path, parse_dates=['ts']) for path in paths]
    print(f"Loaded {len(dfs)} files")
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

    print(df.head())

    return df


if __name__ == "__main__":
    logged_model = 'runs:/18bc1107358f412aa057e855772c136f/ohlcv-xgb-20230721173019'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    data = get_df()
    data = data.iloc[0:60]
    print(data.describe())
    print(data.head())
    print(data.info())
    # min and max index
    print(data.index.min())
    print(data.index.max())

    print(loaded_model.predict(data))

    df_orig = get_df()
    predictions = []
    dfs = []
    n_range = 200
    lag = 60
    tss = []
    for i in range(n_range):
        from_idx = i
        to_idx = from_idx + lag

        prediciton = loaded_model.predict(df_orig.iloc[from_idx:to_idx])
        dfs.append(df_orig.iloc[from_idx:to_idx])
        predictions.append(prediciton)

    df_pred = pd.concat(predictions, axis=0)
    df_pred.rename(
        columns={"prediction_mean": 'close_pred', 'prediction_std': 'close_pred_std', 'prediction_timestamp': 'ts'},
        inplace=True)
    df_pred.set_index('ts', inplace=True)
    print(df_pred.head())

    # outer join on ts as index
    df = df_pred.join(df_orig, how='inner')
    print(df.head())

    df.drop(columns=["open", "high", "low", "volume", "close_pred_std"], inplace=True)
    df.plot()
    plt.show()
