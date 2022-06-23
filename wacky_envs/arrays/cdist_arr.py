import numpy as np
from scipy import spatial

from wacky_envs import ValueEnvModule
from wacky_envs.arrays import BaseArray


class CDistArray(BaseArray):
    """
    Calculate distances between two arrays. See the scipy documentation of
    `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist>`__
    for more information.
    """

    def __init__(self, arr1, arr2, metric='euclidean'):
        """
        Initialize.

        :param arr1: Array with at least two values in last dim.
        :param arr2: Array with at least two values in last dim. Last dim of arr1 and arr2 must be the same.
        :param metric: Kind of distance, see scipy documentation for the full list.
        """
        super(CDistArray, self).__init__()
        self._arr1 = arr1
        self._arr2 = arr2
        self._metric = metric

    @property
    def arr1(self):
        return self._arr1

    @property
    def arr2(self):
        return self._arr2

    @property
    def metric(self):
        return self._metric

    @property
    def value(self):
        return spatial.distance.cdist(
            self.check_dims(self._arr1.value),
            self.check_dims(self._arr2.value),
            metric=self._metric
        ).squeeze()

    @staticmethod
    def check_dims(arr):

        if isinstance(arr, ValueEnvModule):
            arr = arr.value

        if arr.ndim == 1:
            return np.expand_dims(arr, axis=0)
        else:
            return arr
