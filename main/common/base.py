import numpy as np
import pandas as pd
from collections.abc import Iterable
from pandas import Index, RangeIndex

class BaseLearner(object):

    def __init__(self):

        self._causal_matrix = None

    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value

class Tensor(np.ndarray):
    """
    Parameters
    ----------
    object: array-like
        Multiple list, ndarray, DataFrame
    index: Index or array-like
        Index to use for resulting tensor. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns: Index or array-like
        Column labels to use for resulting tensor. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
    """

    def __new__(cls, object=None, index=None, columns=None):

        if object is None:
            raise TypeError("Tensor() missing required argument 'object' (pos 0)")
        elif isinstance(object, list):
            object = np.array(object)
        elif isinstance(object, pd.DataFrame):
            index = object.index
            columns = object.columns
            object = object.values
        elif isinstance(object, (np.ndarray, cls)):
            pass
        else:
            raise TypeError(
                "Type of the required argument 'object' must be array-like."
            )
        if index is None:
            index = range(object.shape[0])
        if columns is None:
            columns = range(object.shape[1])
        obj = np.asarray(object).view(cls)
        obj.index = index
        obj.columns = columns

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if self.ndim == 0: return
        elif self.ndim == 1:
            self.columns = RangeIndex(0, 1, step=1, dtype=int)
        else:
            self.columns = RangeIndex(0, self.shape[1], step=1, dtype=int)
        self.index = RangeIndex(0, self.shape[0], step=1, dtype=int)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, Iterable)
        if len(list(value)) != self.shape[0]:
            raise ValueError("Size of value is not equal to the shape[0].")
        self._index = Index(value)

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        assert isinstance(value, Iterable)
        if (self.ndim > 1 and len(list(value)) != self.shape[1]):
            raise ValueError("Size of value is not equal to the shape[1].")
        self._columns = Index(value)