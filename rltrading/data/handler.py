from typing import List
from datetime import datetime, timedelta
import calendar
import numpy as np
import pandas as pd

from rltrading.data.fundamental import (
    get_social_sentiment,
)
from rltrading.data.meta_trader import get_stock_data


def get_data(
    symbol: str, _from: datetime, to: datetime, lookback: timedelta
) -> pd.DataFrame:
    meta_trader_data = get_stock_data(symbol, _from, to)
    social_sentiment = get_social_sentiment(symbol, _from, to, lookback=lookback)
    times = meta_trader_data["time"].values
    social_sentiment = __aggregate_social_sentiment(social_sentiment, times, lookback)
    total = meta_trader_data.merge(social_sentiment, how="left")
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


if __name__ == "__main__":
    _from = datetime(
        year=2022,
        month=8,
        day=10,
        hour=0,
        minute=0,
        second=0,
    )

    to = datetime(
        year=2022,
        month=12,
        day=31,
        hour=23,
        minute=59,
        second=59,
    )
    lookback = timedelta(days=1)
    get_data("GME", _from, to, lookback)
