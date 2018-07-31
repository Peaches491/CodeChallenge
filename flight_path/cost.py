import numpy as np
import itertools
import logging

log = logging.getLogger(__name__)


def _normalize(x):
    return x / np.linalg.norm(x)


def _pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def compute(flight_plan, wind, cost_func):
    edge_costs = []
    for head, tail in _pairwise(flight_plan):
        edge_costs.append(cost_func(head, tail, wind))
    log.info("Edge segment costs: %s", edge_costs)
    return edge_costs, sum(edge_costs)


def distance(flight_plan):
    return sum(
        np.linalg.norm(tail - head) for head, tail in _pairwise(flight_plan))


def _fixed_cost(head, tail, wind, unit_dist_cost):
    delta = tail - head
    return unit_dist_cost * np.linalg.norm(delta)


def _dot_cost(head, tail, wind, airspeed):
    res = 0.1
    path_length = np.linalg.norm(tail - head)
    r = np.linspace(0, 1, path_length / res)
    points = np.outer(r, (tail - head))
    points += head

    wind_vecs = [wind.at(*p) for p in points]
    zip_direction = _normalize(tail - head)
    zip_vec = zip_direction * airspeed
    cost = 0
    for s in wind_vecs:
        cost += np.linalg.norm(zip_vec - s) * res
    return cost


def make_fixed_cost(airspeed):
    return lambda *args: _fixed_cost(*args, airspeed)


def make_dot_cost(airspeed):
    return lambda *args: _dot_cost(*args, airspeed)
