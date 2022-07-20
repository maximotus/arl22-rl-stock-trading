# script to test connection to a local MetaTrader5 terminal
# using the MetaTrader python module, see https://www.mql5.com/de/docs/integration/python_metatrader5
# make sure you have installed the MetaTrader5 terminal here: https://www.metatrader5.com/en

import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pandas as pd
import pytz

from datetime import datetime
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# connect with MetaTrader5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

print(mt5.terminal_info())
print(mt5.version())

utc = pytz.timezone("Etc/UTC")

# get 1000 ticks of the Nvidia symbol (i.e. NVDA) starting from a certain timestamp (date and time)
nvda_ticks = mt5.copy_ticks_from("NVDA", datetime(2022, 1, 12, 20, tzinfo=utc), 1000, mt5.COPY_TICKS_ALL)

# get all ticks between 27. January 2022 16:30 UTC and 28. January 2022 21:00 UTC of the Google symbol (i.e. GOOGL)
googl_ticks = mt5.copy_ticks_range("GOOGL", datetime(2022, 1, 27, 16, tzinfo=utc),
                                   datetime(2022, 1, 27, 21, tzinfo=utc), mt5.COPY_TICKS_ALL)

# get 10 bars over 4 hours from 12. January 2022 20:00 UTC for both NVDA and GOOGL
nvda_rates = mt5.copy_rates_from("NVDA", mt5.TIMEFRAME_H4, datetime(2022, 1, 12, 20, tzinfo=utc), 10)
googl_rates = mt5.copy_rates_from("GOOGL", mt5.TIMEFRAME_H4, datetime(2022, 1, 12, 20, tzinfo=utc), 10)

# get 1000 bars from position / index 0 over one minute
eurgbp_rates = mt5.copy_rates_from_pos("EURGBP", mt5.TIMEFRAME_M1, 0, 1000)

# get one bar from specified time range
eurcad_rates = mt5.copy_rates_range("EURCAD", mt5.TIMEFRAME_M1, datetime(2020, 1, 27, 13), datetime(2020, 1, 28, 13))

# close connection to MetaTrader5
mt5.shutdown()

# access data
print('nvda_ticks(', len(nvda_ticks), ')')
for val in nvda_ticks[:10]:
    print(val)

print('googl_ticks(', len(googl_ticks), ')')
for val in googl_ticks[:10]:
    print(val)

print('nvda_rates(', len(nvda_rates), ')')
for val in nvda_rates:
    print(val)

print('googl_rates(', len(googl_rates), ')')
for val in googl_rates:
    print(val)

print('eurgbp_rates(', len(eurgbp_rates), ')')
for val in eurgbp_rates[:10]:
    print(val)

print('eurcad_rates(', len(eurcad_rates), ')')
for val in eurcad_rates[:10]:
    print(val)

ticks_frame = pd.DataFrame(nvda_ticks)
print(ticks_frame)

rates_frame = pd.DataFrame(nvda_rates)
print(rates_frame)

ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
plt.plot(ticks_frame['time'], ticks_frame['ask'], 'r-', label='ask')
plt.plot(ticks_frame['time'], ticks_frame['bid'], 'b-', label='bid')
plt.legend(loc='upper left')
plt.title('NVDA ticks')
plt.show()
plt.close()