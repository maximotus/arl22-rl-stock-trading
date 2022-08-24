from datetime import datetime, timedelta
from typing import Iterable, List, Tuple
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

    def observations(self: "Data", columns: List[str] = None) -> Iterable[List[float]]:
        """Iterate over all observations.

        Yields
        ------
        data : List[float]
            The data of the current observation as a list of floats.
        """
        if columns is not None:
            reduced_df = self._data_frame[columns]
        else:
            reduced_df = self._data_frame
        reduced_df = reduced_df.sort_values(["time"])
        for _, value in reduced_df.iterrows():
            yield value.tolist()

    def __len__(self: "Data") -> int:
        """Get the number of observations.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).

        Returns
        -------
        int
            The number of observations.
        """
        return len(self._data_frame)

    @property
    def shape(self: "Data") -> Tuple[int, int]:
        """Get the number of observations and
        the number of data points per observation.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).

        Returns
        -------
        Tuple[int, int]
            Number of observations, Number of datapoints per observation
        """
        return self._data_frame.shape
