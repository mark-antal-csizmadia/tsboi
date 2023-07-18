import logging
from typing import List, Literal
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseController:
    def __init__(
            self,
            database: str,
            user: str,
            password: str,
            host: str,
            port: str):
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def open_connection(
            self) \
            -> None:
        raise NotImplementedError

    def close_connection(
            self) \
            -> None:
        raise NotImplementedError

    def create_ohlcv_table(
            self,
            table_name: str) \
            -> None:
        raise NotImplementedError

    def insert_ohlcv_record(
            self,
            table_name: str,
            ts: str,
            open_value: float,
            high_value: float,
            low_value: float,
            close_value: float,
            volume_value: float,
            decimal_places: int = 6) \
            -> None:
        raise NotImplementedError

    def insert_ohlcv_records(
            self,
            table_name: str,
            tss: List[str],
            open_values: List[float],
            high_values: List[float],
            low_values: List[float],
            close_values: List[float],
            volume_values: List[float],
            decimal_places: int = 6) \
            -> None:
        raise NotImplementedError

    def get_ohlcv_records(
            self,
            table_name: str,
            ts_start: str,
            ts_end: str,
            data_dir: Path,
            granularity: Literal['minute', 'hour'] = 'minute',
            chunk_size: int = 60) \
            -> List[Path]:
        raise NotImplementedError
