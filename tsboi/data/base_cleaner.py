import time
import logging
from typing import List, Literal
import pandas as pd
from pathlib import Path
import dask.bag as db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCleaner:
    def __init__(
            self,
            data_dir_raw: Path,
            data_dir_cleaned: Path):
        self.data_dir_raw = data_dir_raw
        self.data_dir_cleaned = data_dir_cleaned

    @staticmethod
    def load_from_disk(
            path: Path) \
            -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=['ts'])

    @staticmethod
    def save_to_disk(
            df: pd.DataFrame,
            path: Path):
        df.to_csv(path, index=True)

    @staticmethod
    def interpolate_if_missing(
            df: pd.DataFrame) \
            -> pd.DataFrame:
        df["open"] = df["open"].interpolate()
        df["high"] = df["high"].interpolate()
        df["low"] = df["low"].interpolate()
        df["close"] = df["close"].interpolate()
        df["volume"] = df["volume"].interpolate()
        assert df["ts"].isna().sum() == 0, 'ts column has missing values, cannot interpolate'
        return df

    def clean_step(
            self,
            file_name: str) \
            -> None:
        path_load = self.data_dir_raw / (file_name + '.csv')
        path_save = self.data_dir_cleaned / (file_name + '.csv')
        df = self.load_from_disk(path=path_load)
        df = self.interpolate_if_missing(df=df)
        df = df.set_index('ts')
        self.save_to_disk(df=df, path=path_save)

    def clean(
            self,
            file_names: List[str],
            backend: Literal['dask', 'sequential'] = 'dask') \
            -> None:
        if backend == 'sequential':
            start = time.time()
            for file_name in file_names:
                self.clean_step(file_name=file_name)
            end = time.time()

        elif backend == 'dask':
            start = time.time()
            db.from_sequence(file_names).map(self.clean_step).compute()
            end = time.time()
        else:
            raise ValueError(f'backend {backend} not supported')

        logger.info(f'time taken: {end - start:.4f} seconds for {len(file_names)} files (backend: {backend})')
