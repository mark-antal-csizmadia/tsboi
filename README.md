# tsboi

Time-series forecasting of crypto. Fetching data with [cctx](https://github.com/ccxt/ccxt), training models with [Darts](https://github.com/unit8co/darts) and MLOps (model version, deploy, monitor) with [MLflow](https://github.com/mlflow/mlflow).

## Contributing

Install [DVC](https://dvc.org/doc/install) with your preferred method. For instance, on Mac M1/M2

```bash
brew install dvc
```

> DVC is installed on the machine outside of the virtual environment due to the difficulty of installing it on M1/M2 Macs. It is assumed that the virtual environment is created with Miniconda.

Create virtual environment (Python version 3.9 is tested) with preferably [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install dependencies:

```bash
conda create -n tsboi-env python=3.9
conda activate tsboi-env
pip install pip-tools
pip-compile -o requirements.txt requirements.in
pip install -r requirements.txt
```

> For all scripts and tests, it is assumed that there exists a Posgres server running locally on port 5432 with a database named `PG_DATABASE`, a user named `PG_USER` with password `PG_PASSWORD`. These can be set as environment variables or in a `.env` file in the root directory of the project.
> For instance:
> ```bash
> CREATE DATABASE btcusddb;
> CREATE USER testuser;
> alter user testuser with encrypted password 'qwerty';
> grant all privileges on database btcusddb to testuser;
> ```

## Tests

Run tests with:

```bash
python -m pytest -s
```

## Prepare data

### Fetch data and insert into database table

For instance, to fetch 1m data for BTC/USD from Binance and insert into table `ohlcv_data`, ending at 2023-07-01T00:00:00Z, and starting from whatever the exchange lets us see at this time, with chunk size 720 (i.e. fetch only 720 minutes or 12 hours at a time):
```bash
python examples/fetch_data.py --table_name "ohlcv_data" --symbol "BTC/USD" --exchange_name "binance" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "1m" --chunk_size 720
```

### Prepare data for training

For instance, to prepare per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/raw/`):
```bash
 python examples/pull_data.py --data_dir "data/raw" --table_name "ohlcv_data" --from_timestamp "2020-08-12T00:00:00Z" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "minute" --chunk_size 60
```

### Clean data for training

> Cleaning is only interpolating missing values so far. Outlier removal and other cleaning methods will be added later.

For instance, to clean per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/cleaned/`):
```bash
python examples/clean_data.py --data_dir_raw "data/raw" --data_dir_cleaned "data/cleaned"
```

### Split data for training

> Splitting is done by either number of observations in the train, validation and test subsets, or by ratio of all observations in the train, validation and test subsets, or by timestamp splitting of all observations.

For instance, to split per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/split/train`, `data/split/val` and `data/split/test`):
```bash
python examples/split_data.py --data_dir_cleaned "data/cleaned" --data_dir_split "data/split" --ratio --ratio_train 0.8 --ratio_val 0.12
```

### Train

> For distributed hyperparameter tuning (via pyspark) [JDK](https://www.oracle.com/java/technologies/downloads/#jdk20-mac) needs to be installed

#### Simple fit

```bash
python examples/fit_xgb.py --dataset_digest=9c3bf8a86cf74d9f84fd38d86087314e70766a95 --random_state=42
```
where `dataset_digest` is the digest of the dataset in the database, obtained by running
```bash
git log -n 1 --pretty=format:%H -- data.dvc
```

#### Hyperparameter tuning and refit

```bash
python tsboi/hp_search_with_refit_ohlcv_xgb.py
```


### Test

> Replace: `logged_model` with the path to the logged model
```bash
python tsboi/test_ohlcv_xgb.py
```
