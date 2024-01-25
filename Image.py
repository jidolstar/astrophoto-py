import numpy as np
class Image:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Image는 numpy array이어야 합니다.")
        self._data = data

    @property
    def data(self):
        return self._data