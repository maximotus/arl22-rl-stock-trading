import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

import numpy as np
from rltrading import Data
from rltrading.data.data import Config

load_dotenv()
fh_key = os.getenv("FINNHUB_API_KEY", None)

symbol = "AAPL"
start = datetime(2022, 8, 18, 1, 0, 0)
end = datetime(2022, 8, 18, 23, 59, 59)
social_lookback = timedelta(days=1)

data_config = Config(
    symbol=symbol, from_=start, to=end, lookback=social_lookback, finnhub_api_key=fh_key
)
aapl_data = Data()

dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, "data", "example")
aapl_data.fetch(store=True, config=data_config, dir_path=path)
fetched_obs = [observation for observation in aapl_data.observations()]

# Re-Initialize the apple data and laod from the previously created csv file.
aapl_data = Data(
    symbol=symbol,
    from_=start,
    to=end,
    lookback=social_lookback,
    finnhub_api_key=fh_key,
)
aapl_data.load(symbol=data_config.symbol, dir_path=path)
loaded_obs = [observation for observation in aapl_data.observations()]

# print("Fetched data is equal to loaded data: ", np.allclose(fetched_obs, loaded_obs))
print("Length: ", len(aapl_data))
print("Shape: ", aapl_data.shape)

aapl_data.load(symbol=data_config.symbol, dir_path=path)
loaded_obs = [
    observation
    for observation in aapl_data.observations(
        columns=["time", "open", "high", "low", "close"]
    )
]
print(loaded_obs[0])
