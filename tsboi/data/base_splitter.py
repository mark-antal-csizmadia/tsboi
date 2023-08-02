from pathlib import Path
from glob import glob
import logging
import pandas as pd
from typing import Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseSplitter:
    TIMESTAMP_COLUMN_NAME = 'ts'

    def __init__(
            self,
            data_dir_cleaned: Path,
            data_dir_split: Path,
            file_pattern: str = "*.csv"):

        self.data_dir_cleaned = data_dir_cleaned
        self.data_dir_split = data_dir_split
        self.file_pattern = file_pattern

    def load_data_from_disk(
            self) \
            -> pd.DataFrame:

        paths = sorted(glob(str(self.data_dir_cleaned / self.file_pattern)))
        logger.info(f"Loading %s files from %s", len(paths), self.data_dir_cleaned)

        dfs = [pd.read_csv(path, parse_dates=[BaseSplitter.TIMESTAMP_COLUMN_NAME]) for path in paths]
        df = pd.concat(dfs, axis=0)
        logger.info(f"Loaded %s rows", df.shape[0])

        return df

    @staticmethod
    def split_by_timestamps(
            df: pd.DataFrame,
            train_end: pd.Timestamp,
            val_end: pd.Timestamp) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_train = df[df[BaseSplitter.TIMESTAMP_COLUMN_NAME] <= train_end]
        df_val = df[(df[BaseSplitter.TIMESTAMP_COLUMN_NAME] > train_end) & (df[BaseSplitter.TIMESTAMP_COLUMN_NAME] <= val_end)]
        df_test = df[df[BaseSplitter.TIMESTAMP_COLUMN_NAME] > val_end]

        return df_train, df_val, df_test

    @staticmethod
    def split_by_ratio(
            df: pd.DataFrame,
            train_ratio: float,
            val_ratio: float) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_train = df.iloc[:int(len(df) * train_ratio)]
        df_val = df.iloc[int(len(df) * train_ratio):int(len(df) * (train_ratio + val_ratio))]
        df_test = df.iloc[int(len(df) * (train_ratio + val_ratio)):]

        return df_train, df_val, df_test

    @staticmethod
    def split_by_number(
            df: pd.DataFrame,
            train_num: int,
            val_num: int) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_train = df.iloc[:train_num]
        df_val = df.iloc[train_num:train_num + val_num]
        df_test = df.iloc[train_num + val_num:]

        return df_train, df_val, df_test

    def split(
            self,
            split_mode: str,
            **kwargs) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df = self.load_data_from_disk()
        logger.info(f"Splitting n=%s rows by mode=%s", df.shape[0], split_mode)

        if split_mode == "timestamp":
            train_end = kwargs["train_end"]
            val_end = kwargs["val_end"]
            df_train, df_val, df_test = self.split_by_timestamps(df=df, train_end=train_end, val_end=val_end)

        elif split_mode == "ratio":
            train_ratio = kwargs["train_ratio"]
            val_ratio = kwargs["val_ratio"]
            df_train, df_val, df_test = self.split_by_ratio(df=df, train_ratio=train_ratio, val_ratio=val_ratio)

        elif split_mode == "number":
            train_num = kwargs["train_num"]
            val_num = kwargs["val_num"]
            df_train, df_val, df_test = self.split_by_number(df=df, train_num=train_num, val_num=val_num)

        else:
            raise ValueError("Invalid split mode")

        assert df_train.shape[0] + df_val.shape[0] + df_test.shape[0] == df.shape[0], "Splitting error"

        logger.info(f"Train: %s, Val: %s, Test: %s", df_train.shape[0], df_val.shape[0], df_test.shape[0])
        logger.info("train head: \n%s", df_train.head())
        logger.info("val head: \n%s", df_val.head())
        logger.info("test head: \n%s", df_test.head())

        self.save_split_data_to_disk(df_train=df_train, df_val=df_val, df_test=df_test, chunk_size=60)

        return df_train, df_val, df_test

    def save_split_data_to_disk(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            df_test: pd.DataFrame,
            chunk_size: int) \
            -> None:

        logger.info(f"Saving train, val, test data to %s", self.data_dir_split)

        for df, name in zip([df_train, df_val, df_test], ["train", "val", "test"]):
            lengths = []
            data_dir_split_subset = self.data_dir_split / name
            data_dir_split_subset.mkdir(parents=True, exist_ok=True)

            for i in range(0, len(df), chunk_size):
                df_chunk = df.iloc[i:i+chunk_size]
                lengths.append(df_chunk.shape[0])
                ts_min = df_chunk['ts'].min()
                ts_max = df_chunk['ts'].max()
                df_chunk.to_csv(data_dir_split_subset / f"{ts_min}_{ts_max}.csv", index=False)

            logger.info(f"Saved %s data with %s chunks (data points: %s) at %s", name, len(lengths), sum(lengths),
                        data_dir_split_subset)
