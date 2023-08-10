from typing import Dict, Any, Optional, Union
import time
import multiprocessing as mp
import torch
from darts import TimeSeries
from darts.models import TransformerModel
import numpy as np


def transformer_train_function(
        series_dict: Dict[str, TimeSeries],
        covariates_dict: Dict[str, TimeSeries],
        lags_dict: Dict[str, int],
        random_state: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs: Dict[str, Any]) \
        -> TransformerModel:

    series_train: TimeSeries = series_dict['series_train']
    series_val: TimeSeries = series_dict['series_val']
    series_test: TimeSeries = series_dict['series_test']

    covariates: Union[TimeSeries, None] = covariates_dict.get('covariates', None)
    covariates_train: Union[TimeSeries, None] = covariates_dict.get('covariates_train', None)
    covariates_val: Union[TimeSeries, None] = covariates_dict.get('covariates_val', None)

    # these could be hyperparameters, so we need to get them from the kwargs, and then from the lags_dict
    lags: Union[int, np.int64] = kwargs.pop('lags', lags_dict.get('lags', None))
    lags: int = int(lags)

    model = TransformerModel(
        input_chunk_length=lags,
        output_chunk_length=1,
        batch_size=1024,
        n_epochs=kwargs["n_epochs"],
        model_name=model_name if model_name else int(time.time()),
        nr_epochs_val_period=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        random_state=random_state,
        save_checkpoints=True,
        force_reset=True,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-2},
        pl_trainer_kwargs={"accelerator": "auto", "devices": "auto"}
    )

    model.fit(
        series=series_train,
        past_covariates=covariates_train,
        future_covariates=None,
        val_series=series_val,
        val_past_covariates=covariates_val,
        val_future_covariates=None,
        verbose=True,
        num_loader_workers=mp.cpu_count(),
    )

    return model
