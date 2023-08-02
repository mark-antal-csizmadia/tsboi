import sys
import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.data.base_splitter import BaseSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for cleaning OHLCV data for model training.')
    parser.add_argument('--data_dir_cleaned', help='Directory to load clean data from.', type=lambda p: Path(p),
                        required=True, default=Path('data/cleaned'))
    parser.add_argument('--data_dir_split', help='Directory to save split cleaned data to - ready for training',
                        type=lambda p: Path(p), required=True, default=Path('data/split'))
    parser.add_argument('--number', help='Split by number of rows', action='store_true', dest='number')
    parser.add_argument('--ratio', help='Split by ratio of rows', action='store_true', dest='ratio')
    parser.add_argument('--timestamp', help='Split by timestamp', action='store_true', dest='timestamp')

    parser.add_argument('--num_train', required='--number' in sys.argv, type=int)
    parser.add_argument('--num_val', required='--number' in sys.argv, type=int)

    parser.add_argument('--ratio_train', required='--ratio' in sys.argv, type=float)
    parser.add_argument('--ratio_val', required='--ratio' in sys.argv, type=float)

    parser.add_argument('--timestamp_train_end', required='--timestamp' in sys.argv, type=lambda ts: pd.Timestamp(ts))
    parser.add_argument('--timestamp_val_end', required='--timestamp' in sys.argv, type=lambda ts: pd.Timestamp(ts))

    return parser


def main():
    if args.number:
        split_mode = 'number'
        kwargs = {"train_num": args.num_train, "val_num": args.num_val}
    elif args.ratio:
        split_mode = 'ratio'
        kwargs = {"train_ratio": args.ratio_train, "val_ratio": args.ratio_val}
    elif args.timestamp:
        split_mode = 'timestamp'
        kwargs = {"train_end": args.timestamp_train_end, "val_end": args.timestamp_val_end}
    else:
        raise ValueError(f"Invalid split mode")

    # if data dir exists, delete and recreate
    shutil.rmtree(data_dir_split, ignore_errors=True)
    data_dir_split.mkdir(parents=True, exist_ok=True)

    logger.info(f"Splitting data by {split_mode} with kwargs: {kwargs}")

    data_splitter = BaseSplitter(
        data_dir_cleaned=data_dir_cleaned,
        data_dir_split=data_dir_split,
        file_pattern='*.csv',
    )

    df_train, df_val, df_test = data_splitter.split(split_mode=split_mode, **kwargs)


if __name__ == "__main__":
    args = get_parser().parse_args()
    data_dir_cleaned = args.data_dir_cleaned
    data_dir_split = args.data_dir_split

    main()
