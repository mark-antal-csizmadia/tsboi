import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd

from tsboi.data.pg_controller import PostgresController
from settings import PG_DATABASE, PG_USER, PG_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for preparing OHLCV data for model training.')
    parser.add_argument('--data_dir', help='Directory to save data to.', type=lambda p: Path(p), required=True,
                        default=Path('data/raw'))
    parser.add_argument('--table_name', help='Table name to fetch data from.', type=str, required=True)
    parser.add_argument('--from_timestamp', help='Start date for data (ISO 8601).', type=lambda ts: pd.Timestamp(ts),
                        required=True, default="2020-08-12T00:00:00Z")
    parser.add_argument('--end_timestamp', help='End date for data (ISO 8601).', type=lambda ts: pd.Timestamp(ts),
                        required=True, default="2023-07-01T00:00:00Z")
    parser.add_argument('--periodicity', help='Periodicity of data (e.g. minute, hour, etc.)', type=str, required=True,
                        choices=["minute"])
    parser.add_argument('--chunk_size', help='Number of data points to fetch and store as a chunk in a file.',
                        type=int, required=True)
    parser.add_argument('--postgres_host', help='Postgres host.', type=str, required=False, default="localhost",
                        choices=["localhost", "postgres"])
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    data_dir = args.data_dir
    table_name = args.table_name
    from_timestamp = args.from_timestamp
    end_timestamp = args.end_timestamp
    periodicity = args.periodicity
    chunk_size = args.chunk_size
    postgres_host = args.postgres_host

    # if data dir exists, delete and recreate
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # print without +00:00
    if from_timestamp.tzinfo is not None:
        from_timestamp = from_timestamp.tz_convert(None)
    if end_timestamp.tzinfo is not None:
        end_timestamp = end_timestamp.tz_convert(None)

    logger.info("Fetching data from %s to %s ...", from_timestamp, end_timestamp)

    controller = \
        PostgresController(database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD, host=postgres_host, port='5432')
    paths = controller.get_ohlcv_records(
        table_name=table_name,
        # UTC time
        ts_start=f'{from_timestamp.isoformat()}Z',
        ts_end=f'{end_timestamp.isoformat()}Z',
        granularity=periodicity,
        chunk_size=chunk_size,
        data_dir=data_dir
    )
    logger.info("Saved %s files to %s", len(paths), data_dir)
