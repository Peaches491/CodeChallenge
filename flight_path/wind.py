import numpy as np
from scipy import spatial
import logging

log = logging.getLogger(__name__)


class Wind(object):
    def __init__(self, vectors):
        self._vectors = vectors
        self.kd = spatial.KDTree(self.points)
        self.v_at = np.vectorize(lambda x, y: self.at(x, y))
        self.v_magnitude_at = np.vectorize(lambda x, y: self.magnitude_at(x, y))

    def at(self, x, y):
        point = np.array([[x], [y]]).T
        _, nn_index = self.kd.query(point)
        return self.uv[:, nn_index[0]]

    def magnitude_at(self, x, y):
        point = np.array([[x], [y]]).T
        _, nn_index = self.kd.query(point)
        return self.magnitude[nn_index[0]]

    def __len__(self):
        return len(self._vectors)

    def __getitem__(self, s):
        return self._vectors.__getitem__(s)

    @property
    def x(self):
        return self._vectors[:, 0]

    @property
    def y(self):
        return self._vectors[:, 1]

    @property
    def points(self):
        return self._vectors[:, :2]

    @property
    def angle(self):
        return self._vectors[:, 2]

    @property
    def magnitude(self):
        return self._vectors[:, 3]

    @property
    def u(self):
        return np.cos(np.deg2rad(self.angle)) * self.magnitude

    @property
    def v(self):
        return np.sin(np.deg2rad(self.angle)) * self.magnitude

    @property
    def uv(self):
        return np.vstack([self.u, self.v])
