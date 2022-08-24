import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
from pydantic import BaseModel, PrivateAttr

import pytz
import pandas as pd

from rltrading.data.handler import get_data


class Config(BaseModel):
    symbol: Optional[str]
    from_: Optional[datetime]
    to: Optional[datetime]
    lookback: Optional[timedelta]
    finnhub_api_key: Optional[str]


class Observation(BaseModel):
    _data: pd.Series = PrivateAttr()

    def value(self: "Observation", key: str) -> float:
        """_summary_

        Parameters
        ----------
        self : Observation
            _description_
        key : str
            _description_

        Returns
        -------
        float
            _description_
        """
        return self._data[key].item()

    def all(self: "Observation") -> List[float]:
        return self._data.tolist()


class Data(BaseModel):
    _symbol: str = PrivateAttr(default_factory=None)
    _data_frame: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    # _curr_pos: int = PrivateAttr(default_factory=0)

    def fetch(self: "Data", config: Config, dir_path: Optional[str], store: str = True):
        """Fetch the data via the ``MetaTrader5`` and the ``Finnhub API``.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).
        dir_path : str, optional
            Path to where the data should be saved to.
            The path must already exist.
        store : bool, optional
            If the data should be stored to the given ``path``.
        """
        if store and (dir_path is None):
            raise ValueError(
                "The path can not be 'None' if the 'store' "
                + "parameter is set to 'True'"
            )
        self._symbol = config.symbol
        start = pytz.utc.localize(config.from_)
        end = pytz.utc.localize(config.to)
        self._data_frame = get_data(
            fh_key=config.finnhub_api_key,
            symbol=config.symbol,
            _from=start,
            to=end,
            lookback=config.lookback,
        )

        if store and (dir_path is not None):
            path = os.path.join(dir_path, f"{config.symbol}.csv")
            self._data_frame.to_csv(path, index=False)

    def load(self: "Data", symbol: str, dir_path: str):
        """Load the data from a previously fetched ``pd.DataFrame``.

        Parameters
        ----------
        self : Data
            An instance of the Data class (itself).
        path : str
            Path to the ``.csv`` file containing the ``pd.DataFrame``.
        """
        self._symbol = symbol
        full_path = os.path.join(dir_path, f"{symbol}.csv")
        self._data_frame = pd.read_csv(Path(full_path))

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

    def get_attributes(self: "Data") -> List[str]:
        """_summary_

        Parameters
        ----------
        self : Data
            _description_

        Returns
        -------
        List[str]
            _description_
        """
        return self._data_frame.columns.values.tolist()

    def reduce_attributes(self: "Data", selection: List[str]):
        """_summary_

        Parameters
        ----------
        self : Data
            _description_
        selection : List[str]
            _description_
        """
        self._data_frame = self._data_frame[selection]

    def item(self: "Data", time_step: int) -> Observation:
        """_summary_

        Parameters
        ----------
        self : Data
            _description_
        index : int
            _description_

        Returns
        -------
        List[float]
            _description_
        """
        observation = Observation(_data=self._data_frame.iloc[time_step])
        return observation

    def has_next(self: "Data", time_step: int) -> bool:
        """_summary_

        Parameters
        ----------
        self : Data
            _description_
        index : int
            _description_

        Returns
        -------
        bool
            _description_
        """
        return time_step < (len(self) - 1)

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
