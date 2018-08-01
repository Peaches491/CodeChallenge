import numpy as np
from scipy import spatial
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)


class Wind(object):
    def __init__(self, vectors, barycentric=False):
        self._vectors = vectors
        self.kd = spatial.KDTree(self.points)
        self.barycentric = barycentric
        if self.barycentric:
            self.tris = spatial.Delaunay(
                self.points, qhull_options=['Qbb', 'Qc', 'Qz', 'Qt'])
        self.at = self.weighted_at
        self.v_at = np.vectorize(lambda x, y: self.at(x, y))
        self.v_magnitude_at = np.vectorize(lambda x, y: self.magnitude_at(x, y))

    def weighted_at(self, x, y):
        # return self._nearest_weight(x, y)
        return self._distance_weight(x, y, squared=True)
        return self._barycentric_weight(x, y) or (0, 0)
        return self._barycentric_weight(x, y) or self._distance_weight(
            x, y, squared=False)

    def _barycentric_weight(self, x, y):
        num_points = 1
        point = np.array([(x, y)]).T
        tet = self.tris.find_simplex(point[:, 0])
        if tet == -1:
            return None
        log.info("tet %s", tet)
        # X = self.tris.transform[tet, :3]
        # Y = point - self.tris.transform[tet, 2]
        # log.info("X %s", X)
        # log.info("Y %s", Y)
        T = self.tris.transform[tet, :num_points]
        r = point - self.tris.transform[tet, num_points]
        b = T.dot(r)
        c = np.c_[b, 1 - b.sum(axis=1)]
        log.info("c %s", c)
        if (b > 0).all():
            return (100, 0)
        # log.info("c %s", c)

    def _distance_weight(self, x, y, squared=False):
        point = np.array([[x], [y]]).T
        neighbors = len(self)
        nn_dist, nn_indices = self.kd.query(point, k=neighbors)
        weights = 1. / (nn_dist[0]**(2. if squared else 1.))
        weighted_winds = []
        for a, b in zip(self.uv[:, nn_indices[0]], weights):
            weighted_winds.append(b * a)
        weighted_winds = np.array(weighted_winds).reshape(2, neighbors)
        return np.sum(weighted_winds, axis=1) / np.sum(weights)

    def _nearest_weight(self, x, y, squared=False):
        point = np.array([[x], [y]]).T
        nn_dist, nn_indices = self.kd.query(point, k=1)
        return self.uv[:, nn_indices[0]]

    def nn_at(self, x, y):
        point = np.array([[x], [y]]).T
        _, nn_index = self.kd.query(point)
        return self.uv[:, nn_index[0]]

    def magnitude_at(self, x, y):
        return np.linalg.norm(self.at(x, y))
        # point = np.array([[x], [y]]).T
        # _, nn_index = self.kd.query(point)
        # return self.magnitude[nn_index[0]]

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
