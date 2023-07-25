from typing import Dict, Any, Optional, List, Union
from darts import TimeSeries
from darts.models import XGBModel
import numpy as np


def xgb_train_function(
        series_dict: Dict[str, TimeSeries],
        covariates_dict: Dict[str, TimeSeries],
        lags_dict: Dict[str, int],
        probabilistic_dict: Dict[str, Any],
        random_state: Optional[int] = None,
        **kwargs: Dict[str, Any]) \
        -> XGBModel:

    series_train: TimeSeries = series_dict['series_train']
    series_val: TimeSeries = series_dict['series_val']
    series_test: TimeSeries = series_dict['series_test']

    covariates: Union[TimeSeries, None] = covariates_dict.get('covariates', None)
    covariates_train: Union[TimeSeries, None] = covariates_dict.get('covariates_train', None)
    covariates_val: Union[TimeSeries, None] = covariates_dict.get('covariates_val', None)

    # these could be hyperparameters, so we need to get them from the kwargs, and then from the lags_dict
    lags: Union[int, np.int64, List[int], List[np.int64], None] = \
        kwargs.pop('lags', lags_dict.get('lags', None))
    lags_past_covariates: Union[int, np.int64, List[int], List[np.int64], None] = \
        kwargs.pop('lags_past_covariates', lags_dict.get('lags_past_covariates', None))

    # force cast for darts compatibility
    # if list of np.int64, convert to list of int, or if np.int64, convert to int
    if lags:
        lags = [int(lag) for lag in lags] if isinstance(lags, list) else int(lags)
    if lags_past_covariates:
        lags_past_covariates = [int(lag) for lag in lags_past_covariates] \
            if isinstance(lags_past_covariates, list) else int(lags_past_covariates)

    likelihood: Union[str, None] = probabilistic_dict.get('likelihood', None)
    quantiles: Union[List[float], None] = probabilistic_dict.get('quantiles', None)

    model = XGBModel(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=None,
        output_chunk_length=1,
        add_encoders=None,
        likelihood=likelihood,
        quantiles=quantiles,
        random_state=random_state,
        multi_models=True,
        use_static_covariates=True,
        objective='reg:squarederror',
        early_stopping_rounds=10,
        **kwargs
    )

    model.fit(
        series=series_train,
        past_covariates=covariates_train,
        future_covariates=None,
        val_series=series_val,
        val_past_covariates=covariates_val,
        val_future_covariates=None,
        verbose=True
    )

    return model
