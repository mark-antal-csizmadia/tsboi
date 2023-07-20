import sys
import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, AfterValidator

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO
from tsboi.data.base_cleaner import BaseCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for cleaning OHLCV data for model training.')
    parser.add_argument('--data_dir_raw', help='Directory to load raw data from.', type=lambda p: Path(p),
                        required=True, default=Path('data/raw'))
    parser.add_argument('--data_dir_cleaned', help='Directory to save cleaned data to.', type=lambda p: Path(p),
                        required=True, default=Path('data/cleaned'))
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    data_dir_raw = args.data_dir_raw
    data_dir_cleaned = args.data_dir_cleaned

    # if data dir exists, delete and recreate
    shutil.rmtree(data_dir_cleaned, ignore_errors=True)
    data_dir_cleaned.mkdir(parents=True, exist_ok=True)

    paths = [Path(path) for path in list(data_dir_raw.glob('*.csv'))]
    logger.info("Cleaning %s files ...", len(paths))
    cleaner = BaseCleaner(data_dir_raw=data_dir_raw, data_dir_cleaned=data_dir_cleaned)

    filenames = [Path(path).stem for path in paths]
    cleaner.clean(file_names=filenames, backend="dask")
    logger.info("Cleaned %s files, and saved them to %s", len(filenames), data_dir_cleaned)
