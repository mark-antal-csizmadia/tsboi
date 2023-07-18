import logging
from typing import Literal, List, Protocol, Optional
import pandas as pd
import ccxt

from tsboi.data.base_fetcher import Fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaveChunkFunctionType(Protocol):
    def __call__(
            self,
            tss: List[str],
            open_values: List[float],
            high_values: List[float],
            low_values: List[float],
            close_values: List[float],
            volume_values: List[float]) -> str:
        ...


class CCTXFetcher(Fetcher):
    def __init__(
            self,
            exchange_name: Literal["binance"] = "binance"):
        super().__init__(exchange_name=exchange_name)
        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_name)()

    def fetch_ohlcv_data_step(
            self,
            until: pd.Timestamp = pd.Timestamp("2023-06-01T00:00:00Z"),
            symbol: Literal["BTC/USD"] = "BTC/USD",
            timeframe: Literal["1m", "1h", "1d"] = "1m",
            limit: int = 60) \
            -> pd.DataFrame:

        data = self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            params={"until": self.exchange.parse8601(until.isoformat())}
        )
        if len(data) == 0:
            raise ValueError("No data fetched, probably reached limit")
        else:
            logger.info("Fetched data until %s, limit %s", until, limit)

        df = pd.DataFrame(data)
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('ts')
        return df

    def fetch_ohlcv_data(
            self,
            save_chunk_function: SaveChunkFunctionType,
            symbol: Literal["BTC/USD"] = "BTC/USD",
            end: pd.Timestamp = pd.Timestamp("2023-06-01T00:00:00Z"),
            periodicity: Literal["1m", "1h", "1d"] = "1m",
            chunk_size: int = 60,
            break_after_n_chunks: Optional[int] = None) \
            -> int:

        logger.info(f"Fetching data from {self.exchange_name} exchange, symbol {symbol}, periodicity {periodicity}, "
                    f"end {end} (will go back in time as much as possible)")
        end_dt = pd.to_datetime(end)
        # periodicity to timedelta object
        timedelta = pd.to_timedelta(periodicity)
        # calculate datetime at the start of the first desired period
        start_dt = end_dt - chunk_size * timedelta

        number_of_data_points = 0
        while True:
            try:
                df = self.fetch_ohlcv_data_step(
                    until=end_dt,
                    symbol=symbol,
                    timeframe=periodicity,
                    limit=chunk_size
                )

                save_chunk_function(
                    tss=df.index.strftime("%Y-%m-%dT%H:%M:%SZ").values,
                    open_values=df["open"].values,
                    high_values=df["high"].values,
                    low_values=df["low"].values,
                    close_values=df["close"].values,
                    volume_values=df["volume"].values
                )

                number_of_data_points += len(df)
                end_dt = start_dt
                start_dt = end_dt - chunk_size * timedelta

                if break_after_n_chunks:
                    if number_of_data_points >= break_after_n_chunks * chunk_size:
                        logger.info("Reached break_after_n_chunks limit")
                        break
            except ValueError:
                logger.info("No more data to fetch")
                break

        logger.info("Fetched %s data points", number_of_data_points)
        return number_of_data_points
