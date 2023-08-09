from typing import List, Optional, Tuple, Dict
from glob import glob
import logging
from pathlib import Path
from darts import TimeSeries, concatenate
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset:
    TIMESTAMP_COLUMN_NAME = 'ts'

    def __init__(
            self,
            data_dir: Path,
            file_pattern: str,
            target_id: str,
            covariate_ids: Optional[List[str]] = None,
            dtype: Optional[str] = 'float32') \
            -> None:

        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.target_id = target_id
        self.covariate_ids = covariate_ids
        self.dtype = dtype
        self.freq = None
        self.description = None
        self.examples_df = None
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.test_dir = self.data_dir / 'test'
        self.train_paths = sorted(glob(str(self.train_dir / self.file_pattern)))
        self.val_paths = sorted(glob(str(self.val_dir / self.file_pattern)))
        self.test_paths = sorted(glob(str(self.test_dir / self.file_pattern)))

    def load_data_from_disk(
            self,
            subset: Optional[str] = None,
            limit: Optional[int] = None) \
            -> pd.DataFrame:

        if subset == 'train':
            paths = self.train_paths
            logger.info(f"Loading %s files from %s", len(paths), self.train_dir)
        elif subset == 'val':
            paths = self.val_paths
            logger.info(f"Loading %s files from %s", len(paths), self.val_dir)
        elif subset == 'test':
            paths = self.test_paths
            logger.info(f"Loading %s files from %s", len(paths), self.test_dir)
        elif subset == 'train_val':
            paths = self.train_paths + self.val_paths
            logger.info(f"Loading %s files from %s, and %s", len(paths), self.train_dir, self.val_dir)
        else:
            paths = self.train_paths + self.val_paths + self.test_paths
            logger.info(f"Loading %s files from %s, %s, and %s", len(paths),
                        self.train_dir, self.val_dir, self.test_dir)

        # TODO: remove this
        paths = paths[:10]
        dfs = [pd.read_csv(path, parse_dates=[BaseDataset.TIMESTAMP_COLUMN_NAME]) for path in paths]
        df = pd.concat(dfs, axis=0)
        logger.info(f"Loaded %s rows", df.shape[0])

        # drop everything that is not timestamp, target or covariate
        df = df[[BaseDataset.TIMESTAMP_COLUMN_NAME, self.target_id] + self.covariate_ids]

        # limit number of rows
        df = df.iloc[:limit] if limit else df
        logger.info(f"Using %s rows", df.shape[0])

        return df

    def get_series_and_covariates(
            self,
            df: pd.DataFrame) \
            -> Tuple[TimeSeries, TimeSeries]:

        # remove timezone info and set index
        df[BaseDataset.TIMESTAMP_COLUMN_NAME] = df[BaseDataset.TIMESTAMP_COLUMN_NAME].dt.tz_localize(None)
        df.set_index(BaseDataset.TIMESTAMP_COLUMN_NAME, inplace=True)

        # infer frequency from model_input
        freq = pd.infer_freq(df.index)
        assert freq is not None, "Could not infer frequency from data"
        logger.info(f"Inferred frequency of data set is: '%s'", freq)

        # cast to dtype
        df[self.target_id] = df[self.target_id].astype(self.dtype)
        if self.covariate_ids:
            df[self.covariate_ids] = df[self.covariate_ids].astype(self.dtype)

        # create series and covariates
        series = TimeSeries.from_dataframe(df, value_cols=self.target_id, freq=freq)

        if self.covariate_ids:
            covariates_list = []
            for covariate_id in self.covariate_ids:
                covariates_list.append(TimeSeries.from_dataframe(df, value_cols=covariate_id, freq=freq))

            covariates = concatenate(covariates_list, axis="component")

        else:
            covariates = None

        return series, covariates

    def describe_series_and_covariates(
            self,
            series: TimeSeries,
            covariates: TimeSeries) \
            -> Dict[str, Dict[str, str]]:

        series_description = {}
        series_description["start_time"] = series.start_time()
        series_description["end_time"] = series.end_time()
        series_description["freq"] = series.freq
        series_description["duration"] = series.duration
        series_description["dtype"] = series.dtype
        series_description["n_timesteps"] = series.n_timesteps

        covariates_description = {}
        if covariates:
            covariates_description["start_time"] = covariates.start_time()
            covariates_description["end_time"] = covariates.end_time()
            covariates_description["freq"] = covariates.freq
            covariates_description["duration"] = covariates.duration
            covariates_description["dtype"] = covariates.dtype
            covariates_description["n_timesteps"] = covariates.n_timesteps

        description = \
            {"series": series_description, "covariates": covariates_description} \
            if covariates else {"series": series_description}

        return description

    def load_dataset(
            self,
            limit: Optional[int] = None,
            subset: Optional[str] = None,
            record_description: bool = False,
            record_examples_df_n_timesteps: Optional[int] = None) \
            -> Tuple[TimeSeries, TimeSeries]:

        df = self.load_data_from_disk(subset=subset, limit=limit)
        series, covariates = self.get_series_and_covariates(df=df)
        self.description = \
            self.describe_series_and_covariates(series=series, covariates=covariates) if record_description else None
        self.examples_df = df.reset_index().head(record_examples_df_n_timesteps) if record_examples_df_n_timesteps else None

        return series, covariates
