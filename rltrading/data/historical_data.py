import math

from rltrading.data.utils import create_api_client
from datetime import datetime, timedelta
from candlestick_chart import Candle, Chart

ISIN = "US0378331005"

client = create_api_client()


today = datetime.now()
year_to_date = today - timedelta(days=365)

bucket_size = 59
buckets = 365 // bucket_size

candles = []
for i in range(buckets):
    start = year_to_date + timedelta(days=i * bucket_size)
    end = start + timedelta(days=bucket_size)

    response = client.market_data.ohlc.get(
        period="m1", isin=ISIN, mic=None, from_=start, to=end
    )
    for x in response.results:
        candles.append(
            Candle(
                open=x.o,
                close=x.c,
                high=x.h,
                low=x.l,
                volume=x.v,
                timestamp=x.t.timestamp(),
            )
        )

chart = Chart(candles=candles, title="ISIN: " + ISIN)
chart.set_name(ISIN)
chart.draw()

client.market_data.venues.get