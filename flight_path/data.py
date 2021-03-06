import numpy as np

SQUARE_FLIGHT_PLAN = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (0, 0),
])

SAMPLE_FLIGHT_PLAN = np.array([
    (2, 1),
    (3, 2),
    (4, 2),
    (5.5, 2.5),
    (6.5, 3.5),
    (7.25, 5),
    (6, 5.5),
    (4.5, 4),
    (3, 3),
    (1, 2),
])

NO_WIND = np.array([
    (0.5, 0.5, 0, 0),
])

NORTH_WIND = np.array([
    (0.0, 0.0, 0, 1),
    (1.0, 1.0, 0, 1),
])

FAVORABLE_WIND = np.array([
    (0.5, 0.0, 0., 1.),
    (1.0, 0.5, 90., 1.),
    (0.5, 1.0, 180., 1.),
    (0.0, 0.5, 270., 1.),
])

OPPOSING_WIND = np.array([
    (0.5, 0.0, 180., 1.),
    (1.0, 0.5, 270., 1.),
    (0.5, 1.0, 0., 1.),
    (0.0, 0.5, 90., 1.),
])

SAMPLE_WIND= np.array([
    (4.2, 0.8, 160, 17),
    (5.4, 3.2, 90, 9),
    (7.4, 5.8, 100, 2),
    (4.8, 6.0, 130, 15),
    (3.3, 4.5, 135, 22),
    (0, 3, 175, 11),
])
