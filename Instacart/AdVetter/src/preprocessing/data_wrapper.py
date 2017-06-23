import pandas as pd
import numpy as np

from enum import Enum


class DataWrapper(object):
    """Class to wrap the concatenated data.
    The different Mode settings change the output of test_data and train_data.

    The main advantage of this wrapper is, to write the cleaning, feature engineering and modelling code
    using ``DataWrapper.data``, ``DataWrapper.train_data`` and ``DataWrapper.test_data``.
    For validation and submission, only the ``Mode`` has to be changed, the remaining code remains as it is.

    Examples
    --------
        >>> d = DataWrapper(data_tr, data_te, DataWrapper.Mode.TRAIN)
        >>> d.train_data # data_tr
        >>> d.test_data # data_te

        >>> # data_su is not used in VALIDATION mode
        >>> d = DataWrapper(data_tr, data_te, data_va, data_su, DataWrapper.Mode.VALIDATE)
        >>> d.train_data # data_tr + data_te
        >>> d.test_data # data_va

    """

    class Mode(Enum):
        TRAIN = 1
        VALIDATE = 2
        SUBMIT = 3

    def __init__(self, train, test, validate=None, submit=None, mode=Mode.TRAIN):
        self.__mode = mode

        if self.__mode == self.Mode.TRAIN:
            self.test_offset = len(train)
            data_used = [train, test]
        elif self.__mode == self.Mode.VALIDATE:
            self.test_offset = len(train) + len(test)
            data_used = [train, test, validate]
        elif self.__mode == self.Mode.SUBMIT:
            self.test_offset = len(train) + len(test) + len(validate)
            data_used = [train, test, validate, submit]
        else:
            raise Exception("Please select a predefined Mode from DataWrapper.Mode")

        self.data = pd.concat(data_used, ignore_index=True)

    @property
    def train_data(self):
        return self.data[0:self.test_offset]

    @train_data.setter
    def train_data(self, d):
        if len(d) != self.test_offset:
            raise Exception("Data size does not match")

        self.data[0:self.test_offset] = d

    @property
    def test_data(self):
        return self.data[self.test_offset:]

    @test_data.setter
    def test_data(self, d):
        if len(d) != len(self.data) - self.test_offset:
            raise Exception("Data size does not match")

        self.data[self.test_offset:] = d
