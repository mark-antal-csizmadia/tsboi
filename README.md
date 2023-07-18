# tsboi

Time-series forecasting of crypto with [Darts](https://github.com/unit8co/darts) and MLOps with [MLflow](https://github.com/mlflow/mlflow).

## Contributing

Create virtual environment (Python version 3.9 is tested) with preferably [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install dependencies:

```bash
conda create -n tsboi-env python=3.9
conda activate tsboi-env
pip install pip-tools
pip-compile -o requirements.txt requirements.in
pip install -r requirements.txt
```

For all scripts and tests, it is assumed that there exists a Posgres server running locally on port 5432 with a database named `PG_DATABASE`, a user named `PG_USER` with password `PG_PASSWORD`. These can be set as environment variables or in a `.env` file in the root directory of the project.

## Tests

Run tests with:

```bash
python -m pytest -s
```
