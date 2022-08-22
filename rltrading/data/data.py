from datetime import datetime, timedelta
from typing import Iterable, List
from pydantic import BaseModel, PrivateAttr

import pytz
import pandas as pd

from rltrading.data.handler import get_data


class Data(BaseModel):
    symbol: str
    from_: datetime
    to: datetime
    lookback: timedelta
    finnhub_api_key: str

    _data_frame: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    def fetch(self: "Data", store: str = True, path: str = None):
        """Fetch the data via the ``MetaTrader5`` and the ``Finnhub API``.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).
        path : str, optional
            Path to where the data should be saved to.
        store : bool, optional
            If the data should be stored to the given ``path``.
        """
        if store and (path is None):
            raise ValueError(
                "The path can not be 'None' if the 'store' "
                + "parameter is set to 'True'"
            )

        start = pytz.utc.localize(self.from_)
        end = pytz.utc.localize(self.to)
        self._data_frame = get_data(
            self.finnhub_api_key,
            symbol=self.symbol,
            _from=start,
            to=end,
            lookback=self.lookback,
        )

        if store and (path is not None):
            self._data_frame.to_csv(path, index=False)

    def load(self: "Data", path: str):
        """Load the data from a previously fetched ``pd.DataFrame``.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).
        path : str
            Path to the ``.csv`` file containing the ``pd.DataFrame``.
        """
        self._data_frame = pd.read_csv(path)

    def observations(self: "Data") -> Iterable[List[float]]:
        """Iterate over all observations.

        Yields
        ------
        data : List[float]
            The data of the current observation as a list of floats.
        """
        self._data_frame = self._data_frame.sort_values(["time"])
        for _, value in self._data_frame.iterrows():
            yield value.tolist()
