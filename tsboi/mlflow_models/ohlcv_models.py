import mlflow
from pathlib import Path
import pandas as pd


class MLflowXGBOHLCVModel(mlflow.pyfunc.PythonModel):
    def load_context(
            self,
            context: mlflow.pyfunc.PythonModelContext) \
            -> None:

        from darts.models import XGBModel
        import joblib

        self.path_to_model: Path = \
            context.artifacts["path_to_model_file"]
        self.path_to_pipeline_target: Path = \
            context.artifacts.get("path_to_pipeline_target_file", None)
        self.path_to_pipeline_past_covariates: Path = \
            context.artifacts.get("path_to_pipeline_past_covariates_file", None)

        self.pipeline_target = \
            joblib.load(self.path_to_pipeline_target) if self.path_to_pipeline_target else None
        self.pipeline_past_covariates = \
            joblib.load(self.path_to_pipeline_past_covariates) if self.path_to_pipeline_past_covariates else None

        self.model: XGBModel = XGBModel.load(str(self.path_to_model))

    def format_inputs(
            self,
            model_input: pd.DataFrame):

        from darts import TimeSeries
        from darts import concatenate

        # infer frequency from model_input
        freq = pd.infer_freq(model_input.index)
        assert freq is not None, "Could not infer frequency from model_input"
        # print(f"freq: {freq}")

        series = TimeSeries.from_dataframe(model_input, value_cols="close", freq=freq)
        covariate_open = TimeSeries.from_dataframe(model_input, value_cols="open", freq=freq)
        covariate_high = TimeSeries.from_dataframe(model_input, value_cols="high", freq=freq)
        covariate_low = TimeSeries.from_dataframe(model_input, value_cols="low", freq=freq)
        # covariate_volume = TimeSeries.from_dataframe(model_input, value_cols="volume", freq=FREQ)

        covariates = concatenate([covariate_open, covariate_high, covariate_low], axis="component")

        if self.pipeline_target is not None:
            series = self.pipeline_target.transform(series)

        if self.pipeline_past_covariates is not None:
            covariates = self.pipeline_past_covariates.transform(covariates)

        return series, covariates

    def format_outputs(
            self, outputs) \
            -> pd.DataFrame:

        prediction_mean = outputs.mean().values()[0][0]
        prediction_std = outputs.std().values()[0][0] if outputs.is_stochastic else 0
        prediction_timestamp = outputs.time_index[0]

        return pd.DataFrame({
            "prediction_mean": [prediction_mean],
            "prediction_std": [prediction_std],
            "prediction_timestamp": [prediction_timestamp]
        })

    def predict(self, context, model_input):
        from darts import TimeSeries

        series, covariates = self.format_inputs(model_input=model_input)

        prediction: TimeSeries = self.model.predict(
            n=1,
            series=series,
            past_covariates=covariates,
            num_samples=1,
        )

        # inverse transform
        if self.pipeline_target is not None:
            prediction = self.pipeline_target.inverse_transform(prediction, partial=True)

        return self.format_outputs(outputs=prediction)
