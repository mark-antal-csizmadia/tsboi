import os
import logging
from typing import List, Literal
import psycopg2
import pandas as pd
from pathlib import Path

from tsboi.data.base_controller import BaseController


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresController(BaseController):
    def __init__(
            self,
            database: str,
            user: str,
            password: str,
            host: str,
            port: str):

        super().__init__(
            database=database,
            user=user,
            password=password,
            host=host,
            port=port)
        self.conn = None
        self.cursor = None

    def open_connection(
            self) \
            -> None:

        logger.debug(f"Opening connection to {self.host}:{self.port}")
        self.conn = psycopg2.connect(
            database=self.database, user=self.user, password=self.password, host=self.host, port=self.port)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def close_connection(
            self) \
            -> None:

        logger.debug(f"Opening connection to {self.host}:{self.port}")
        self.conn.close()

    def create_ohlcv_table(
            self,
            table_name: str) \
            -> None:

        self.open_connection()
        logger.debug(f"Creating table {table_name} in database {self.database}")
        self.cursor.execute("DROP TABLE IF EXISTS " + table_name)
        sql = '''CREATE TABLE ''' + table_name + '''(
            ts TIMESTAMP WITH TIME ZONE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT
            )'''
        self.cursor.execute(sql)
        self.close_connection()

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

        self.open_connection()
        logger.debug(f"Inserting record into table {table_name} in database {self.database}")
        sql = '''INSERT INTO ''' + table_name + '''(
            ts,
            open,
            high,
            low,
            close,
            volume
            ) VALUES (
            TIMESTAMP WITH TIME ZONE ' ''' + ts + ''' ',
            ''' + str(round(open_value, decimal_places)) + ''',
            ''' + str(round(high_value, decimal_places)) + ''',
            ''' + str(round(low_value, decimal_places)) + ''',
            ''' + str(round(close_value, decimal_places)) + ''',
            ''' + str(round(volume_value, decimal_places)) + '''
            )'''
        self.cursor.execute(sql)
        self.close_connection()

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

        for ts, open_value, high_value, low_value, close_value, volume_value in \
                zip(tss, open_values, high_values, low_values, close_values, volume_values):
            self.insert_ohlcv_record(
                table_name=table_name,
                ts=ts,
                open_value=open_value,
                high_value=high_value,
                low_value=low_value,
                close_value=close_value,
                volume_value=volume_value,
                decimal_places=decimal_places
            )

    def get_ohlcv_records(
            self,
            table_name: str,
            ts_start: str,
            ts_end: str,
            data_dir: Path,
            granularity: Literal['minute', 'hour'] = 'minute',
            chunk_size: int = 60) \
            -> List[Path]:

        self.open_connection()
        sql = f"""
            WITH aggregated_data AS (
                SELECT 
                    DATE_TRUNC('{granularity}', ts) AS ts_truncated,
                    (array_agg(open ORDER BY ts ASC))[1] AS first_open,
                    MAX(high) AS max_high,
                    MIN(low) AS min_low,
                    (array_agg(close ORDER BY ts DESC))[1] AS last_close,
                    SUM(volume) AS sum_volume
                FROM {table_name}
                WHERE ts >= TIMESTAMP WITH TIME ZONE '{ts_start}'
                AND ts <= TIMESTAMP WITH TIME ZONE '{ts_end}'
                GROUP BY ts_truncated
            )
            SELECT 
                ad.ts_truncated,
                ad.first_open,
                ad.max_high,
                ad.min_low,
                ad.last_close,
                ad.sum_volume
            FROM aggregated_data ad
            ORDER BY ad.ts_truncated ASC
        """

        # how many records to buffer on a client
        self.cursor.itersize = 1000
        self.cursor.execute(sql)

        os.makedirs(data_dir, exist_ok=True)

        rows = []
        paths = []
        for row in self.cursor:
            rows.append(row)
            if len(rows) % chunk_size == 0:
                df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                # convert to timestamp
                df['ts'] = pd.to_datetime(df['ts'])
                # localise to UTC
                df['ts'] = df['ts'].dt.tz_convert('UTC')
                ts_min = df['ts'].min()
                ts_max = df['ts'].max()
                df = df.set_index('ts')
                path = data_dir / f'{ts_min}_{ts_max}.csv'

                df.to_csv(path)
                rows = []
                paths.append(path)

        self.close_connection()

        return paths
