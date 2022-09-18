from typing import List
from datetime import datetime, timedelta
import calendar
import time
import numpy as np
import pandas as pd

from rltrading.data.fundamental import (
    get_social_sentiment,
    get_social_sentiment_alternative,
)
from rltrading.data.meta_trader import get_asset_data


def get_data(
    fh_key: str, symbol: str, _from: datetime, to: datetime, lookback: timedelta
) -> pd.DataFrame:
    print("Getting asset data...")
    start = time.time()
    meta_trader_data = get_asset_data(symbol, _from, to)
    print(f"Successfully got asset data. ({time.time() - start})")
    print("Getting social sentiment...")
    start = time.time()
    days = __get_days(meta_trader_data["time"])
    social_sentiment = get_social_sentiment_alternative(
        fh_key, symbol, days, lookback=lookback
    )
    print(f"Successfully got socidal sentiment. ({time.time() - start})")
    times = meta_trader_data["time"].values
    print("Aggregating social sentiment...")
    start = time.time()
    social_sentiment = __aggregate_social_sentiment(social_sentiment, times, lookback)
    print(f"Successfully aggregated social sentiment. ({time.time() - start})")
    print("Mergin data...")
    start = time.time()
    total = meta_trader_data.merge(social_sentiment, how="left")
    print(f"Merged data. ({time.time() - start})")
    total = total.fillna(value=0.0)
    total = total.sort_values(["time"])
    return total


def __aggregate_social_sentiment(
    sentiments: pd.DataFrame, times: List[int], lookback: timedelta
) -> pd.DataFrame:
    sentiments = __presort_sentiments(sentiments, times, lookback)
    intermediates = []
    for upper_bound in times:
        lower_bound = __substract_time(upper_bound, lookback)
        intermediate = sentiments.loc[
            (sentiments["time"] >= lower_bound) & (sentiments["time"] <= upper_bound)
        ].copy(deep=True)
        if len(intermediate) == 0:
            continue
        intermediate["time"] = upper_bound
        intermediates.append(intermediate)
    total_sentiments = pd.concat(intermediates)
    total_sentiments = total_sentiments.groupby(["time"]).describe().reset_index()
    total_sentiments.columns = [
        "_".join(col) if col[1] != "" else col[0]
        for col in total_sentiments.columns.values
    ]
    return total_sentiments


def __presort_sentiments(
    sentiments: pd.DataFrame, times: List[int], lookback: timedelta
) -> pd.DataFrame:
    total_lower_bound = __substract_time(np.min(times), lookback)
    bigger_than_min = sentiments["time"] >= total_lower_bound
    smaller_than_max = sentiments["time"] <= np.max(times)
    sentiments = sentiments.loc[bigger_than_min & smaller_than_max]
    return sentiments


def __substract_time(timestamp: int, lookback: timedelta) -> int:
    return calendar.timegm(
        (datetime.fromtimestamp(timestamp) - lookback).utctimetuple()
    )


def __get_days(times: List[int]):
    dates = list(map(__to_date, times))
    unique_days = np.unique(dates).tolist()
    return unique_days


def __to_date(time: int) -> datetime:
    dt = datetime.fromtimestamp(time)
    return datetime(dt.year, dt.month, dt.day)
