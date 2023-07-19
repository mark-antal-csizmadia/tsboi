from pathlib import Path
import shutil
import pytest

from tsboi.data.pg_controller import PostgresController
from settings import PG_DATABASE, PG_USER, PG_PASSWORD


@pytest.fixture
def pg_controller():
    return PostgresController(
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        host='localhost',
        port='5432')


@pytest.fixture
def test_data():
    return [
        ('2021-01-01 00:00:00+00:00', 1.0, 2.0, 0.5, 1.5, 100.0),
        ('2021-01-01 00:01:00+00:00', 1.5, 2.5, 1.0, 2.0, 200.0),
        ('2021-01-01 00:02:00+00:00', 2.0, 3.0, 1.5, 2.5, 300.0),
        ('2021-01-02 00:01:00+00:00', 2.5, 3.5, 2.0, 3.0, 400.0),
        ('2021-01-02 00:02:00+00:00', 3.0, 4.0, 2.5, 3.5, 500.0),
        ('2021-01-02 00:03:00+00:00', 3.5, 4.5, 3.0, 4.0, 600.0),
        ('2021-01-03 00:01:00+00:00', 4.0, 5.0, 3.5, 4.5, 700.0),
        ('2021-01-03 00:02:00+00:00', 4.5, 5.5, 4.0, 5.0, 800.0),
        ('2021-01-03 00:03:00+00:00', 5.0, 6.0, 4.5, 5.5, 900.0),
        ('2021-01-03 00:04:00+00:00', 5.5, 6.5, 5.0, 6.0, 1000.0)]


def test_open_connection(pg_controller):
    pg_controller.open_connection()
    assert pg_controller.conn.closed == 0


def test_close_connection(pg_controller):
    pg_controller.open_connection()
    pg_controller.close_connection()
    assert pg_controller.conn.closed == 1


def test_create_ohlcv_table(pg_controller):
    pg_controller.create_ohlcv_table(table_name="test_table")

    pg_controller.open_connection()
    # check if table exists
    pg_controller.cursor.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='test_table')")
    assert pg_controller.cursor.fetchone()[0]

    # assert schema
    pg_controller.cursor.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'test_table'")
    assert pg_controller.cursor.fetchall() == [
        ('ts', 'timestamp with time zone'),
        ('open', 'double precision'),
        ('high', 'double precision'),
        ('low', 'double precision'),
        ('close', 'double precision'),
        ('volume', 'double precision')]

    # clean up
    pg_controller.cursor.execute("DROP TABLE test_table")
    pg_controller.close_connection()


def test_insert_ohlcv_record(pg_controller, test_data):
    pg_controller.create_ohlcv_table(table_name="test_table")
    pg_controller.insert_ohlcv_record(
        table_name="test_table",
        ts=test_data[0][0],
        open_value=test_data[0][1],
        high_value=test_data[0][2],
        low_value=test_data[0][3],
        close_value=test_data[0][4],
        volume_value=test_data[0][5])

    # check if record exists
    pg_controller.open_connection()
    pg_controller.cursor.execute(
        f"SELECT EXISTS (SELECT FROM test_table WHERE ts='{test_data[0][0]}')")
    # check if record exists
    assert pg_controller.cursor.fetchone()[0]

    # clean up
    pg_controller.cursor.execute("DROP TABLE test_table")
    pg_controller.close_connection()


def test_insert_ohlcv_records(pg_controller, test_data):
    pg_controller.create_ohlcv_table(table_name="test_table")
    pg_controller.insert_ohlcv_records(
        table_name="test_table",
        tss=[test_data[idx][0] for idx in range(len(test_data))],
        open_values=[test_data[idx][1] for idx in range(len(test_data))],
        high_values=[test_data[idx][2] for idx in range(len(test_data))],
        low_values=[test_data[idx][3] for idx in range(len(test_data))],
        close_values=[test_data[idx][4] for idx in range(len(test_data))],
        volume_values=[test_data[idx][5] for idx in range(len(test_data))])

    # check if records exist
    pg_controller.open_connection()
    for idx in range(len(test_data)):
        pg_controller.cursor.execute(
            f"SELECT EXISTS (SELECT FROM test_table WHERE ts='{test_data[idx][0]}')")
        assert pg_controller.cursor.fetchone()[0]

    # clean up
    pg_controller.cursor.execute("DROP TABLE test_table")
    pg_controller.close_connection()


def test_get_ohlcv_records(pg_controller, test_data):
    pg_controller.create_ohlcv_table(table_name="test_table")
    pg_controller.insert_ohlcv_records(
        table_name="test_table",
        tss=[test_data[idx][0] for idx in range(len(test_data))],
        open_values=[test_data[idx][1] for idx in range(len(test_data))],
        high_values=[test_data[idx][2] for idx in range(len(test_data))],
        low_values=[test_data[idx][3] for idx in range(len(test_data))],
        close_values=[test_data[idx][4] for idx in range(len(test_data))],
        volume_values=[test_data[idx][5] for idx in range(len(test_data))])

    paths = pg_controller.get_ohlcv_records(
        table_name="test_table",
        ts_start=test_data[0][0],
        ts_end=test_data[-1][0],
        data_dir=Path("test_data"),
        granularity="minute",
        chunk_size=5)

    assert len(paths) == int(len(test_data) // 5)
    # clean up
    pg_controller.open_connection()
    pg_controller.cursor.execute("DROP TABLE test_table")
    pg_controller.close_connection()
    shutil.rmtree("test_data")
