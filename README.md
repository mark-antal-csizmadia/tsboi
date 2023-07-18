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
