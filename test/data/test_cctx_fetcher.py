from typing import List
import pandas as pd


def test_fetch_ohlcv_data(fetcher, pg_controller):

    def save_chunk_function(
            tss: List[str],
            open_values: List[float],
            high_values: List[float],
            low_values: List[float],
            close_values: List[float],
            volume_values: List[float]) \
            -> str:

        pg_controller.insert_ohlcv_records(
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

    table_name = "test_table"
    pg_controller.create_ohlcv_table(table_name=table_name)

    number_of_data_points = fetcher.fetch_ohlcv_data(
        save_chunk_function=save_chunk_function,
        symbol="BTC/USD",
        end=pd.Timestamp("2023-07-01T00:00:00Z"),
        periodicity="1m",
        chunk_size=60*12,
        break_after_n_chunks=2
    )

    assert number_of_data_points == 60*12*2

    pg_controller.open_connection()
    pg_controller.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    number_of_records = pg_controller.cursor.fetchone()[0]
    assert number_of_records == number_of_data_points

    pg_controller.cursor.execute(f"DROP TABLE {table_name};")
    pg_controller.close_connection()
