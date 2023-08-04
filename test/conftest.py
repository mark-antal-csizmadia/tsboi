import pytest

from tsboi.data.pg_controller import PostgresController
from tsboi.data.cctx_fetcher import CCTXFetcher
from examples.settings import PG_DATABASE, PG_USER, PG_PASSWORD


def pytest_addoption(parser):
    parser.addoption('--postgres_host', action='store', default='localhost',
                     help='Postgres host (need to be container name in docker-compose)')


@pytest.fixture
def pg_controller(request):
    return PostgresController(
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        host=request.config.getoption('--postgres_host'),
        port='5432')


@pytest.fixture
def fetcher():
    return CCTXFetcher("binance")
