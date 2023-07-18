import logging
from typing import Literal, Callable, List
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Fetcher:
    def __init__(
            self,
            exchange_name: Literal["binance"] = "binance"):
        self.exchange_name = exchange_name

    def fetch_ohlcv_data_step(
            self,
            until: pd.Timestamp = pd.Timestamp("2023-06-01T00:00:00Z"),
            symbol: Literal["BTC/USD"] = "BTC/USD",
            timeframe: Literal["1m", "1h", "1d"] = "1m",
            limit: int = 60) \
            -> pd.DataFrame:
        """ Fetches one chunk of OHLCV data. This is a step in the process of fetching all the data.

        Parameters
        ----------
        until : pd.Timestamp
            Time stamp to fetch data until. Defaults to "2023-06-01T00:00:00Z". Needs to be in ISO 8601 format.
        symbol : str
            Symbol to fetch data for. Defaults to "BTC/USD".
        timeframe : str
            Periodicity of the data to fetch. Defaults to "1m". Any Pandas frequency string is accepted.
        limit : int
            Number of rows to fetch in one request. Defaults to 60. This depends on the exchange.

        Returns
        -------
        pd.DataFrame
            Data frame with the fetched data. Should have columns "ts", "open", "high", "low", "close" and "volume".
            "ts" should be a datetime column, and the rest should be floats. "ts" should be the index of the data frame
            upon return.
        """

        raise NotImplementedError

    def fetch_ohlcv_data(
            self,
            save_chunk_function: Callable[
                [List[str], List[float], List[float], List[float], List[float], List[float]], str],
            symbol: Literal["BTC/USD"] = "BTC/USD",
            end: pd.Timestamp = pd.Timestamp("2023-06-01T00:00:00Z"),
            periodicity: Literal["1m", "1h", "1d"] = "1m",
            chunk_size: int = 60) \
            -> int:
        """ Fetches OHLCV data from exchange and saves it to a table in the database.

        Parameters
        ----------
        save_chunk_function : Callable[[List[str], List[float], List[float], List[float], List[float], List[float]], str]
            Function to save fetched chunk of data. Could be a function that saves to a database or to a file. Accepts
            the timestamp, open, high, low, close and volume as lists, and returns a string with some reference to the
            saved data.
        symbol : str
            Symbol to fetch data for. Defaults to "BTC/USD".
        end : pd.Timestamp
            Time stamp to fetch data until. Defaults to "2023-06-01T00:00:00Z". Needs to be in ISO 8601 format.
        periodicity : str
            Periodicity of the data to fetch. Defaults to "1m". Any Pandas frequency string is accepted.
        chunk_size : int
            Number of rows to fetch in one request. Defaults to 60. This depends on the exchange.

        Returns
        -------
        int
            Number of data points fetched.
        """

        raise NotImplementedError
