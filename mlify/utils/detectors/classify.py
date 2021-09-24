import os
import enum

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from pandas import DataFrame
from typing import Any, Optional, Tuple


class MissingNullFillingValue(Exception):
    pass


class RemoveNullBy(enum.Enum):
    dropping: str = "drop"
    filling: str = "fill"
    imputer: str = "imputer"


class Detector(object):
    def __init__(self, filename: str) -> None:
        super().__init__()

        assert os.path.exists(filename), "The file does not exist"

        self.df: Optional[DataFrame] = None

        self.get_df(filename)

    def get_df(self, filename: str) -> None:
        ext: Tuple[str, str] = os.path.splitext(filename)

        if ext == ".csv":
            self.df = pd.read_csv(filename)
        elif ext == ".xlsx":
            self.df = pd.read_excel(filename)
        else:
            raise NotImplementedError(f"The extension {ext} does not work.")

    def optimise_for_null(
        self, action: RemoveNullBy, fillby: Optional[Any] = None, missing_values=np.nan
    ) -> None:

        if self.df is not None and self.df.isnull().any().any() is True:
            if action.dropping:
                self.df = self.df.dropna()
            else:
                if fillby is None:
                    raise MissingNullFillingValue(
                        "Please add a filling value using the `fillBy` parameter."
                    )
                if action.filling:
                    self.df = self.df.fillna(fillby)
                elif action.imputer:
                    imputer = SimpleImputer(
                        missing_values=missing_values, strategy=fillby
                    )
                    self.df = imputer.fit_transform(self.df)
