from __future__ import annotations
from datetime import datetime, time, timedelta
from enum import Enum
import os
from time import time
from typing import Tuple
from black import main

import plotext as plt
import pandas as pd

from lemon import api
from dotenv import load_dotenv

import plotly.graph_objects as go


class IntraDayOptions(Enum):
    HOUR = 0
    MINUTE = 1


class Order:
    def __init__(self, client, order_id) -> Order:
        self.__client = client
        self.__order_id = order_id

    def execute(self):
        self.__client.trading.orders.activate(order_id=self.__order_id)

    def cancel(self):
        self.__client.trading.orders.cancel(order_id=self.__order_id)


class Lemon:
    def __init__(self) -> Lemon:
        load_dotenv()
        self.__client = api.create(
            market_data_api_token=os.getenv("MARKET_DATA_API_TOKEN", None),
            trading_api_token=os.getenv("TRADING_API_TOKEN", None),
            env="paper",
        )

    def buy(self, isin: str, quantity: str) -> Order:
        response = self.__client.trading.orders.create(
            isin=isin,
            side="buy",
            quantity=quantity,
        )
        order_id = response.results.id
        return Order(self.__client, order_id)

    def sell(self, isin: str, quantity: str) -> Order:
        response = self.__client.trading.orders.create(
            isin=isin,
            side="sell",
            quantity=quantity,
        )
        order_id = response.results.id
        return Order(self.__client, order_id)


    def plot_intraday(self, isin: str, date: datetime, tick_opts: IntraDayOptions):
        start, end = self.__get_opening_hours()
        start_dt = datetime(date.year, date.month, date.day, start.hour, start.minute)
        end_dt = datetime(date.year, date.month, date.day, end.hour, end.minute)

        response = self.__client.market_data.ohlc.get(
            period="h1" if IntraDayOptions.HOUR == tick_opts else "m1",
            isin=isin,
            mic="XMUN",
            from_=start_dt,
            to=end_dt,
        )
        ohlcs = self.__ohlc_to_pandas(response.results)
        plt.candlestick(ohlcs.timestamp, ohlcs)
        plt.title(isin)
        plt.show()

    def show_hour_chart(self, isin: str, from_: datetime, to: datetime, live: bool = False):
        pass

    def show_day_chart(self, isin: str, from_: datetime, to: datetime, live: bool = False):
        # start_dt = datetime(date.year, date.month, date.day, start.hour, start.minute)
        # end_dt = datetime(date.year, date.month, date.day, end.hour, end.minute)

        response = self.__client.market_data.ohlc.get(
            period="d1",
            isin=isin,
            mic="XMUN",
            from_=from_,
            to=to,
        )
        ohlcs = self.__ohlc_to_pandas(response.results)
        dates = plt.datetimes_to_string(ohlcs.timestamp)
        fig = go.Figure(data=[go.Candlestick(x=dates, open=ohlcs.Open, high=ohlcs.High, low=ohlcs.Low, close=ohlcs.Close)])
        fig.show()
        # plt.candlestick(dates, ohlcs)
        # plt.title(isin)
        # plt.show()


    def __load_data(self, isin: str, from_: datetime, to: datetime):
        pass

    def __get_opening_hours(self) -> Tuple[time, time]:
        response = self.__client.market_data.venues.get(mic='XMUN')
        open_hours = response.results[0].opening_hours
        return open_hours.start, open_hours.end
    
    def __ohlc_to_pandas(self, ohlcs):
        data = []
        for ohlc in ohlcs:
            data.append({
                "Open": ohlc.o,
                "Close": ohlc.c,
                "High": ohlc.h,
                "Low": ohlc.l,
                "Volume": ohlc.v,
                "timestamp": ohlc.t,
            })
        return pd.DataFrame(data)



if __name__ == '__main__':
    l = Lemon()
    # l.plot_intraday("US0378331005", datetime(2022, 8, 12), IntraDayOptions.MINUTE)
    today = datetime.now()
    then = today - timedelta(days=59)
    l.show_day_chart("US0378331005", then, today)
