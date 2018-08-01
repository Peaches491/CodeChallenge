#!/usr/bin/env python3

import numpy as np

points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])

from scipy.spatial import Delaunay
tri = Delaunay(points)

# We can plot it:
import matplotlib.pyplot as plt
plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o')
# plt.show()

# Point indices and coordinates for the two triangles forming the triangulation:
tri.simplices
points[tri.simplices]

# We can find out which triangle points are in:
p = np.array([
    (0.1, 0.2),
    (0.1, 0.9),
    # (1.5, 0.5),
]).T
print(tri.find_simplex(p))

# We can also compute barycentric coordinates in triangle 1 for these points:
T = tri.transform[1, :2]
print("T.shape: %s", T.shape)
x = tri.transform[1, 2]
print("x.shape: %s", x.shape)
y = (p - x)
print("y.shape: %s", y.shape)
b = T.dot(y)
print("b.shape: %s", b.shape)
print(np.c_[b, 1 - b.sum(axis=1)])
