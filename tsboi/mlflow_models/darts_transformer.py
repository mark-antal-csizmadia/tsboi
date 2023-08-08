import logging

import mlflow
from pathlib import Path
import pandas as pd
from typing import Union, Optional


class MLflowDartsTransformerModel(mlflow.pyfunc.PythonModel):
    def load_context(
            self,
            context: mlflow.pyfunc.PythonModelContext) \
            -> None:

        import joblib
        import yaml
        from darts.models import TransformerModel
        from darts.dataprocessing.pipeline import Pipeline
        from torch.cuda import is_available as torch_cuda_is_available
        from torch.backends.mps import is_available as torch_mps_is_available

        self.path_to_model: str = context.artifacts["path_to_model_file"]
        logging.info("Loading model from: {}".format(self.path_to_model))

        self.path_to_model_info_file: str = context.artifacts["path_to_model_info_file"]
        self.path_to_model_info_file: Path = Path(self.path_to_model_info_file)

        self.path_to_pipeline_target: Optional[str] = context.artifacts.get("path_to_pipeline_target_file", None)
        self.path_to_pipeline_target: Optional[Path] = \
            Path(self.path_to_pipeline_target) if self.path_to_pipeline_target else None

        self.path_to_pipeline_past_covariates: Optional[str] = \
            context.artifacts.get("path_to_pipeline_past_covariates_file", None)
        self.path_to_pipeline_past_covariates: Optional[Path] = \
            Path(self.path_to_pipeline_past_covariates) if self.path_to_pipeline_past_covariates else None

        self.pipeline_target: Optional[Pipeline] = \
            joblib.load(self.path_to_pipeline_target) if self.path_to_pipeline_target else None

        self.pipeline_past_covariates: Optional[Pipeline] = \
            joblib.load(self.path_to_pipeline_past_covariates) if self.path_to_pipeline_past_covariates else None

        try:
            self.model: TransformerModel = TransformerModel.load(path=self.path_to_model)
            logging.info("Loaded model to GPU.")
        except RuntimeError as e:
            self.model: TransformerModel = TransformerModel.load(path=self.path_to_model, map_location="cpu")
            self.model.to_cpu()
            logging.info("Failed to load model to GPU, loaded model to CPU (reason: {}).".format(e))

        with open(self.path_to_model_info_file, "r") as stream:
            try:
                model_info = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

        self.target_id: str = model_info["target_id"]
        self.covariate_ids: Union[str, None] = model_info.get("covariate_ids", None)
        self.num_samples: int = model_info.get("num_samples", 1)
        if self.num_samples > 1:
            logging.warning("num_samples is 1 when transformer model is used.")
            self.num_samples = 1

        logging.info("Loaded pipeline_target: {}".format(self.pipeline_target))
        logging.info("Loaded pipeline_past_covariates: {}".format(self.pipeline_past_covariates))

    def format_inputs(
            self,
            model_input: pd.DataFrame):

        from darts import TimeSeries
        from darts import concatenate

        model_input.set_index("ts", inplace=True)
        # infer frequency from model_input
        freq = pd.infer_freq(model_input.index)
        assert freq is not None, "Could not infer frequency from model_input"
        # print(f"freq: {freq}")

        # convert target to float32
        model_input[self.target_id] = model_input[self.target_id].astype("float32")

        series = TimeSeries.from_dataframe(model_input, value_cols=self.target_id, freq=freq)

        if self.covariate_ids:
            covariates_list = []
            for covariate_id in self.covariate_ids:
                model_input[covariate_id] = model_input[covariate_id].astype("float32")
                covariate = TimeSeries.from_dataframe(model_input, value_cols=covariate_id, freq=freq)
                covariates_list.append(covariate)

            covariates = concatenate(covariates_list, axis="component")

        else:
            covariates = None

        if self.pipeline_target:
            series = self.pipeline_target.transform(series)

        if self.pipeline_past_covariates:
            covariates = self.pipeline_past_covariates.transform(covariates)

        return series, covariates

    def format_outputs(
            self, outputs) \
            -> pd.DataFrame:

        prediction_mean = outputs.mean().values()[0][0]
        prediction_std = outputs.std().values()[0][0] if outputs.is_stochastic else 0
        prediction_timestamp = outputs.time_index[0]

        return pd.DataFrame({
            "ts": [prediction_timestamp],
            f"{self.target_id}_mean": [prediction_mean],
            f"{self.target_id}_std": [prediction_std]
        })

    def predict(
            self,
            context,
            model_input):

        from darts import TimeSeries

        series, covariates = self.format_inputs(model_input=model_input)

        prediction: TimeSeries = self.model.predict(
            n=1,
            series=series,
            past_covariates=covariates,
            num_samples=self.num_samples,
        )

        # inverse transform
        if self.pipeline_target:
            prediction = self.pipeline_target.inverse_transform(prediction, partial=True)

        return self.format_outputs(outputs=prediction)
