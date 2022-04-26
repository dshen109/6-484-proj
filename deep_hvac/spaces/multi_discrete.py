from __future__ import annotations
from typing import Sequence

import numpy as np

from gym.spaces import multi_discrete


class MultiDiscrete(multi_discrete.MultiDiscrete):
    """
    Extension of gym.spaces.multi_discrete.MultiDiscrete class that allows
    the values to be shifted.
    """

    def __init__(self, starts: None, **kwargs):
        """
        starts: Optional list to shift the range at the corresponding
            index.
        """
        if starts is not None:
            starts = np.array(
                starts, dtype=kwargs.get('dtype', np.int64), copy=True)
        self.starts = starts
        super().__init__(**kwargs)

    def sample(self):
        sampled = super().sample()
        if self.starts is not None:
            sampled += self.starts
        return sampled

    def contains(self, x):
        if isinstance(x, Sequence):
            x = np.array(x)
        if self.starts is not None:
            x = np.array(x) - self.starts
        return super().contains(x)
