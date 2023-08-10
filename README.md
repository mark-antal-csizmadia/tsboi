# tsboi

**t**ime-**s**eries **boi** is a crypto exchange price forecasting package, designed as a ML platform proof-of-concept for the NeurAI project. **tsboi** uses [cctx](https://github.com/ccxt/ccxt) to fetching crypto exchange data, [Darts](https://github.com/unit8co/darts) to train model, [hyperopt](https://github.com/hyperopt/hyperopt) to execute hyperparameter searches, [DVC](https://github.com/iterative/dvc) to version control data, and [MLflow](https://github.com/mlflow/mlflow) to do MLOps (model versioning, deployment, monitoring). 

## Run stuff with `docker-compose`

Run tests:
```bash
docker-compose -f docker-compose-first.yaml up -d --build tests
```

Train an XGB model:
```bash
docker-compose -f docker-compose-first.yaml up -d --build fit_xgb
```

Do hyperparameter tuning for an XGB model:
```bash
docker-compose -f docker-compose-first.yaml up -d --build search_and_refit_xgb_optuna
```

## Contributing

### Install DVC

Install [DVC](https://dvc.org/doc/install) with your preferred method. For instance, on Mac M1/M2

```bash
brew install dvc
```

> DVC is installed on the machine outside of the virtual environment due to the difficulty of installing it on M1/M2 Macs. It is assumed that the virtual environment is created with Miniconda.

### Install requirements

#### Via the `tsboi` package

Create virtual environment (Python version 3.9 is tested) with preferably [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install dependencies:
```bash
conda create -n tsboi-env python=3.9
conda activate tsboi-env
python setup.py install
```

#### Via requirements.txt

Create virtual environment (Python version 3.9 is tested) with preferably [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install dependencies:

```bash
conda create -n tsboi-env python=3.9
conda activate tsboi-env
pip install pip-tools
pip-compile -o requirements.txt requirements.in
pip install -r requirements.txt
```

### Database

> For all scripts and tests, it is assumed that there exists a Posgres server running locally on port 5432 with a database named `PG_DATABASE`, a user named `PG_USER` with password `PG_PASSWORD`. These can be set as environment variables or in a `.env` file in the root directory of the project.
> For instance:
> ```bash
> CREATE DATABASE btcusddb;
> CREATE USER testuser;
> alter user testuser with encrypted password 'qwerty';
> grant all privileges on database btcusddb to testuser;
> ```

### Tests

Run tests with:

```bash
python -m pytest -s
```

## Workflows

### Preparing data

#### Fetch OHLCV data and insert into database table

For instance, to fetch 1m data for BTC/USD from Binance and insert into table `ohlcv_data`, ending at 2023-07-01T00:00:00Z, and starting from whatever the exchange lets us see at this time, with chunk size 720 (i.e. fetch only 720 minutes or 12 hours at a time):
```bash
python examples/fetch_data.py --table_name "ohlcv_data" --symbol "BTC/USD" --exchange_name "binance" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "1m" --chunk_size 720
```

#### Preparing OHLCV data locally for training

For instance, to prepare per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/raw/`):
```bash
 python examples/pull_data.py --data_dir "data/raw" --table_name "ohlcv_data" --from_timestamp "2020-08-12T00:00:00Z" --end_timestamp "2023-07-01T00:00:00Z" --periodicity "minute" --chunk_size 60
```

#### Cleaning OHLCV data for training

> Cleaning is only interpolating missing values so far. Outlier removal and other cleaning methods will be added later.

For instance, to clean per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/cleaned/`):
```bash
python examples/clean_data.py --data_dir_raw "data/raw" --data_dir_cleaned "data/cleaned"
```

#### Splitting OHLCV data for training

> Splitting is done by either number of observations in the train, validation and test subsets, or by ratio of all observations in the train, validation and test subsets, or by timestamp splitting of all observations.

For instance, to split per minute data for training, for BTC/USD from Binance, ending at 2023-07-01T00:00:00Z, and starting from 2020-08-12T00:00:00Z, with chunk size 60 (i.e. in each of the files, there will be 60 minutes of data, so in overall this will be 25272 files at `data/split/train`, `data/split/val` and `data/split/test`):
```bash
python examples/split_data.py --data_dir_cleaned "data/cleaned" --data_dir_split "data/split" --ratio --ratio_train 0.8 --ratio_val 0.12
```

### Training models

> For distributed hyperparameter tuning (via pyspark) [JDK](https://www.oracle.com/java/technologies/downloads/#jdk20-mac) needs to be installed
> For now, only past covariate and past target lags are supported, and the lags parameters can only be integer numbers (i.e.: consider all of the past 10 lags via 10, or all of the past 60 lags via 60).

#### Simple model fit


##### XGBoost

```bash
python examples/fit_xgb.py --dataset_digest=9c3bf8a86cf74d9f84fd38d86087314e70766a95 --random_state=42
```
where `dataset_digest` is the digest of the dataset (the latest commit hash when the dataset was changed), obtained by running
```bash
git log -n 1 --pretty=format:%H -- data.dvc
```

##### Transformer

```bash
python examples/fit_transformer.py --dataset_digest=9c3bf8a86cf74d9f84fd38d86087314e70766a95 --random_state=42
```
where `dataset_digest` is the digest of the dataset (the latest commit hash when the dataset was changed), obtained by running
```bash
git log -n 1 --pretty=format:%H -- data.dvc
```

#### Hyperparameter tuning and best model refit

```bash
python tsboi/hp_search_with_refit_ohlcv_xgb.py
```


#### Inference with model

Load model locally from MlFlow saved model:
```bash
python examples/inference_model.py --logged_model "runs:/6096ca92a6e140f2a48a929b6291bc11/ohlcv-xgb-20230807162915" --n_timesteps 100 --lag_past_covariates 60
```

Deploy model via docker as a REST API (see later, might need to change port number) and then run inference:
```bash
python examples/inference_model.py --n_timesteps 100 --lag_past_covariates 60
```

### Deployment

The models can be deployed as REST API via docker. The docker image can be generated either via implicit or explicit Dockerfile. See the REST API documentation [here](https://mlflow.org/docs/latest/models.html#id66).

#### Compact via implicit Dockerfile

Generate docker image, for instance, for a model run with run_id `runs:/04e248c6acca40bf815203c2ef71dd9e/ohlcv-transformer-20230808152504`:
```bash
mlflow models build-docker --model-uri "runs:/04e248c6acca40bf815203c2ef71dd9e/ohlcv-transformer-20230808152504" --name ohlcv_transformer
docker run -p 5001:8080 ohlcv_transformer
```
The Mlflow server can be accessed at `http://localhost:5001/`.

#### Complex via explicit Dockerfile

Generate docker file and build image, for instance, for registered model `ohlcv_xgb_5` in Production stage:
```bash
mlflow models generate-dockerfile --model-uri "models:/ohlcv_xgb/Production" -d ohlcv_xgb
cd ohlcv_xgb
docker buildx build --platform linux/amd64 --rm -f Dockerfile -t ohlcv_xgb .
docker run -p 5001:8080 ohlcv_xgb
```
The Mlflow server can be accessed at `http://localhost:5001/`.