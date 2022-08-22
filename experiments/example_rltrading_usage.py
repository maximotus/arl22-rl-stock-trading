from datetime import datetime, timedelta

from rltrading import get_data


symbol = "AAPL"
start = datetime(2022, 8, 18, 1, 0, 0)
end = datetime(2022, 8, 18, 23, 59, 59)
social_lookback = timedelta(days=1)

result = get_data(symbol=symbol, _from=start, to=end, lookback=social_lookback)
print(result)
