from datetime import datetime
from typing import List, Tuple

import pytz
import numpy as np
import pandas as pd

import MetaTrader5 as mt5

from tqdm import tqdm


# _DEFAULT_FROM = datetime(
#     year=2018,
#     month=1,
#     day=1,
#     hour=0,
#     minute=0,
#     second=0,
# )
_DEFAULT_FROM = datetime(
    year=2022,
    month=8,
    day=17,
    hour=0,
    minute=0,
    second=0,
)

_DEFAULT_TO = datetime(
    year=2022,
    month=12,
    day=31,
    hour=23,
    minute=59,
    second=59,
)


def get_stock_data(
    symbol: str, from_: datetime = _DEFAULT_FROM, to: datetime = _DEFAULT_TO
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    symbol : str
        _description_
    from_ : datetime, optional
        _description_, by default _DEFAULT_FROM
    to : datetime, optional
        _description_, by default _DEFAULT_TO

    Returns
    -------
    pd.DataFrame
        _description_
    """
    __init_mt5()

    trading_days = __get_trading_days(symbol, from_, to)

    ohlc_df = __get_ohlc_data(symbol, trading_days)
    times = ohlc_df["time"].tolist()

    tick_lookback = __get_tick_lookback(times)
    ticks_df = __get_tick_data(symbol, tick_lookback, times)

    merged = ohlc_df.merge(ticks_df)
    return merged


def __get_ohlc_data(
    symbol: str, trading_days: List[Tuple[datetime, datetime]]
) -> pd.DataFrame:
    ohlc_data = []
    ohlc_columns = []
    for start, end in tqdm(trading_days):
        start = pytz.utc.localize(start)
        end = pytz.utc.localize(end)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)

        ohlc_columns = pd.DataFrame(rates).columns.tolist()
        ohlc_data.extend(list(map(list, rates)))

    ohlc_df = pd.DataFrame(ohlc_data, columns=ohlc_columns).drop_duplicates()
    return ohlc_df


def __get_tick_data(symbol: str, tick_lookback: int, stamps: int) -> pd.DataFrame:
    data_frames = []
    for time in tqdm(stamps):
        ticks = mt5.copy_ticks_range(
            symbol, time - tick_lookback, time, mt5.COPY_TICKS_ALL
        )
        ticks_df = pd.DataFrame(ticks)
        ticks_df["time"] = time
        data_frames.append(ticks_df)

    ticks_df = pd.concat(data_frames)
    ticks_df = ticks_df.groupby(["time"]).describe().reset_index()
    ticks_df.columns = [
        "_".join(col) if col[1] != "" else col[0] for col in ticks_df.columns.values
    ]
    return ticks_df


def __get_tick_lookback(times) -> int:
    return int(np.argmax(np.bincount(np.diff(times))))


def __get_trading_days(
    symbol, from_: datetime, to: datetime
) -> List[Tuple[datetime, datetime]]:
    trading_days = __get_raw_trading_days(symbol, from_, to)

    start_end = []
    for day in trading_days:
        start = datetime(day.year, day.month, day.day, 0, 0, 1)
        end = datetime(day.year, day.month, day.day, 23, 59, 59)
        start_end.append((start, end))
    return start_end


def __get_raw_trading_days(symbol: str, from_: datetime, to: datetime) -> List[int]:
    from_ = pytz.utc.localize(from_)
    to = pytz.utc.localize(to)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H12, from_, to)
    trading_days = rates["time"]
    trading_days = np.unique(list(map(__int_to_date, trading_days)))
    sorted(trading_days)
    return trading_days


def __int_to_date(stamp) -> datetime:
    dt = datetime.fromtimestamp(stamp)
    return datetime(dt.year, dt.month, dt.day)


def __init_mt5():
    if not mt5.initialize():
        raise (RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}"))
