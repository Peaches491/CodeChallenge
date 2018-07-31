import numpy as np
import numpy.testing as npt
import sys
import unittest

import cost
import data
from wind import Wind


class TestCost(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(cost.distance(data.SQUARE_FLIGHT_PLAN), 4)


class TestWindVector(unittest.TestCase):
    def setUp(self):
        import logging
        fmt = "[%(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s"
        logging.basicConfig(level=logging.INFO, format=fmt, file=sys.stdout)

    def test_opposing_wind(self):
        wind = Wind(data.OPPOSING_WIND)
        npt.assert_almost_equal(wind.at(0.0, 0.5), [0.0, 1.0])  # exact
        npt.assert_almost_equal(wind.at(1.0, 0.5), [0.0, -1.0])  # exact
        # Nearest wind vector lies along segment 0
        npt.assert_almost_equal(wind.at(0.1, 0.0), [-1.0, 0.0])


class TestWindMagnitude(unittest.TestCase):
    def setUp(self):
        import logging
        fmt = "[%(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s"
        logging.basicConfig(level=logging.INFO, format=fmt, file=sys.stdout)

    def test_no_wind(self):
        wind = Wind(data.NO_WIND)
        npt.assert_equal(wind.magnitude_at(0.0, 0.0), 0.0)
        npt.assert_equal(wind.magnitude_at(1.0, 1.0), 0.0)

    def test_exact_match(self):
        wind = Wind(data.NO_WIND)
        npt.assert_equal(wind.magnitude_at(0.5, 0.5), 0.0)

    def test_single_wind(self):
        wind = Wind(data.NORTH_WIND)
        npt.assert_equal(wind.magnitude_at(0.0, 0.0), 1.0)
        npt.assert_equal(wind.magnitude_at(0.5, 0.5), 1.0)
        npt.assert_equal(wind.magnitude_at(1.0, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
