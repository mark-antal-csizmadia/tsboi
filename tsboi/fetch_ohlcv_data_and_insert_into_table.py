import sys
import argparse
import logging
from typing import List
import pandas as pd

# TODO: remove this when the code is packaged
sys.path.insert(0, '../tsboi')
# END TODO

from tsboi.data.pg_controller import PostgresController
from tsboi.data.cctx_fetcher import CCTXFetcher
from settings import PG_USER, PG_DATABASE, PG_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PG_CONTROLLER = PostgresController(
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        host='localhost',
        port='5432')


def get_parser() \
        -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='Arguments for fetching OHLCV data, and inserting it into a table.')
    parser.add_argument('--table_name', help='Table name to insert data into.', type=str, required=True)
    parser.add_argument('--symbol', help='Symbol to fetch data for.', type=str, required=True, choices=["BTC/USD"])
    parser.add_argument('--exchange_name', help='Exchange to fetch data from.', type=str, required=True,
                        choices=["binance"])
    parser.add_argument('--end_timestamp', help='End date for data (ISO 8601).', type=lambda ts: pd.Timestamp(ts),
                        required=True, default="2023-07-01T00:00:00Z")
    parser.add_argument('--periodicity', help='Periodicity of data (any pandas periodicity)', type=str, required=True,
                        choices=["1m"])
    parser.add_argument('--chunk_size', help='Number of data points to fetch at a time. Addresses exchange API limits.',
                        type=int, required=True)
    return parser


def save_chunk_function(
        tss: List[str],
        open_values: List[float],
        high_values: List[float],
        low_values: List[float],
        close_values: List[float],
        volume_values: List[float]) \
        -> str:

    PG_CONTROLLER.insert_ohlcv_records(
        table_name=table_name,
        tss=tss,
        open_values=open_values,
        high_values=high_values,
        low_values=low_values,
        close_values=close_values,
        volume_values=volume_values,
        decimal_places=8
    )

    return table_name


def main():

    logger.info(f"Fetching data for {symbol} from {exchange_name} exchange, and inserting it into table {table_name}.")

    fetcher = CCTXFetcher(exchange_name=exchange_name)
    PG_CONTROLLER.create_ohlcv_table(table_name=table_name)

    number_of_data_points = fetcher.fetch_ohlcv_data(
        save_chunk_function=save_chunk_function,
        symbol=symbol,
        end=end_timestamp,
        periodicity=periodicity,
        chunk_size=chunk_size,
        break_after_n_chunks=None
    )

    logger.info(f"Fetched {number_of_data_points} data points, and inserted them into table {table_name}.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    table_name = args.table_name
    symbol = args.symbol
    exchange_name = args.exchange_name
    end_timestamp = args.end_timestamp
    periodicity = args.periodicity
    chunk_size = args.chunk_size

    main()
