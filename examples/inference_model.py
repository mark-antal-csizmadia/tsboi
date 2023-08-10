import argparse
import logging
import mlflow
from pathlib import Path
from glob import glob
import requests
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path('data/split/test')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for cleaning OHLCV data for model training.')
    parser.add_argument('--logged_model', help='MLflow logged model URI', type=str, required=False)
    parser.add_argument('--n_timesteps', help='Number of timesteps to predict', type=int, required=False, default=100)
    parser.add_argument('--lag_past_covariates', help='Number of past covariate lags to use', type=int, required=True)

    return parser


def get_df():
    paths = sorted(glob(str(DATA_DIR / '*.csv')))
    logger.info("Loading %s files from %s", len(paths), DATA_DIR)
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs, axis=0)
    logger.info("Loaded %s rows", df.shape[0])

    return df


def main():
    if logged_model:
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)
    else:
        # Model is running as a server (e.g. in a Docker container)
        host = "127.0.0.1"
        port = "5001"
        url_base = f"http://{host}:{port}"

        url_ping = f"{url_base}/ping"
        url_invocations = f"{url_base}/invocations"

        response = requests.get(url_ping)
        # print code and content
        print(f"response code:\n{response.status_code}")
        print(f"response content:\n{response.content}")
        if response.status_code != 200:
            raise Exception("Server is not running")

    df = get_df()
    print(df.info())
    logger.info("df start ts: %s", df.iloc[0]['ts'])
    logger.info("df end ts: %s", df.iloc[0]['ts'])
    logger.info("lag_past_covariates: %s", lag_past_covariates)

    predictions = []
    dfs = []

    for i in range(n_timesteps):
        from_idx = i
        to_idx = from_idx + lag_past_covariates
        logger.info(f"Predicting next 1 timestep at {df.iloc[to_idx]['ts']}, "
                    f"with covariates/target lags from {df.iloc[from_idx]['ts']} to {df.iloc[to_idx]['ts']}")

        df_context = df.iloc[from_idx:to_idx]

        if logged_model:
            prediction = loaded_model.predict(df.iloc[from_idx:to_idx])
        else:
            json_data = {"dataframe_split": df_context.to_dict(orient="split")}
            response = requests.post(url_invocations, json=json_data)
            prediction = response.json()['predictions']
            prediction = pd.DataFrame(prediction, columns=['ts', 'close_mean', 'close_std'])
            prediction['ts'] = prediction['ts'].astype('datetime64[ns]')

        dfs.append(df_context)
        predictions.append(prediction)

    df_pred = pd.concat(predictions, axis=0)
    df_pred.set_index('ts', inplace=True)
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logged_model = args.logged_model
    lag_past_covariates = args.lag_past_covariates
    n_timesteps = args.n_timesteps
    main()
