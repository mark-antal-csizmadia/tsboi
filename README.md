# tsboi

Time-series forecasting of crypto. Fetching data with [cctx](https://github.com/ccxt/ccxt), training models with [Darts](https://github.com/unit8co/darts) and MLOps (model version, deploy, monitor) with [MLflow](https://github.com/mlflow/mlflow).

## Contributing

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

For instance, to fetch 1m data for BTC/USD from Binance and insert into table `ohlcv_data_1m`, ending at 2023-07-01T00:00:00Z, and starting from whatever the exchange lets us see at this time, with chunk size 720 (i.e. fetch only 720 minutes or 12 hours at a time):
```bash
python tsboi/fetch_ohlcv_data_and_insert_into_table.py --table_name "ohlcv_data_1m" --symbol "BTC/USD" --exchange_name "binance" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "1m" --chunk_size 720
```

### Prepare data for training

For instance, to prepare per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/raw/`):
```bash
 python tsboi/prepare_ohlcv_data.py --data_dir "data/raw" --table_name "ohlcv_data" --from_timestamp "2020-08-12T00:00:00Z" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "minute" --chunk_size 60
```

### Clean data for training

> Cleaning is only interpolating missing values so far. Outlier removal and other cleaning methods will be added later.

For instance, to clean per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/cleaned/`):
```bash
python tsboi/clean_ohlcv_data.py --data_dir_raw "data/raw" --data_dir_cleaned "data/cleaned"
```
