import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

from rltrading import Data
from rltrading import Config

load_dotenv()
fh_key = os.getenv("FINNHUB_API_KEY", None)

symbol = "AAPL"
start = datetime(2018, 1, 1, 0, 0, 1)
end = datetime(2022, 9, 2, 23, 59, 59)
social_lookback = timedelta(days=1)

data_config = Config(
    symbol=symbol, from_=start, to=end, lookback=social_lookback, finnhub_api_key=fh_key
)
aapl_data = Data()

dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, "data", "eval")
start = time.time()
aapl_data.fetch(store=True, config=data_config, dir_path=path)
total = time.time() - start
print(f"DONE ({total})")

# nur trainings daten normalisieren
