#!/usr/bin/env python3
'''
flight_plan - Estimate battery usage of aircraft through wind vector field
'''

import logging

import matplotlib.pyplot as plt
import numpy as np

import cost
import data
import wind as wind_utils

log = logging.getLogger(__name__)


def plot_scene(ax, flight_plan, wind, margin=1):
    log.info("Plotting wind speeds")
    total_dist = cost.distance(flight_plan)

    plan_x = flight_plan[:, 0]
    plan_y = flight_plan[:, 1]

    res = 200
    phi_m = np.linspace(min(plan_x) - margin, max(plan_x) + margin, res)
    phi_p = np.linspace(min(plan_y) - margin, max(plan_y) + margin, res)
    grid_x, grid_y = np.meshgrid(phi_m, phi_p)
    wind_field_mag = wind.v_magnitude_at(grid_x.ravel(), grid_y.ravel())
    wind_field_mag = wind_field_mag.reshape(grid_x.shape)

    if wind.barycentric:
        ax.triplot(wind.x, wind.y, wind.tris.simplices.copy())
        ax.plot(wind.x, wind.y, 'o')


    # Print plots in z render order
    ax.set_aspect('equal')
    ax.pcolormesh(grid_x, grid_y, wind_field_mag)
    ax.scatter(plan_x, plan_y, color='r')
    ax.plot(plan_x, plan_y, color='r')
    ax.scatter(wind.x, wind.y, color='b')
    ax.quiver(wind.x, wind.y, wind.u, wind.v)

    ax.set_title("Waypoints: {} Distance: {:0.4f}".format(
        len(flight_plan), total_dist))
    log.info(ax.title.get_text())


def plot_cost(ax, flight_plan, wind, cost_func):
    log.info("Plotting cost")
    edge_costs, total_cost = cost.compute(flight_plan, wind, cost_func)
    ax.set_title("Total cost: {:0.4f}".format(total_cost))
    ax.bar(np.arange(0, len(edge_costs)), edge_costs, color='r')
    print(ax.title.get_text())


def process(flight_plan, wind_vectors, cost_func, barycentric=False):
    fig, plots = plt.subplots(ncols=2, figsize=(12, 6))

    wind = wind_utils.Wind(wind_vectors, barycentric=barycentric)

    plot_scene(plots[0], flight_plan, wind, margin=2)
    plot_cost(plots[1], flight_plan, wind, cost_func)


def main(argv=None):
    airspeed = 5
    cost_fn = cost.make_dot_cost(airspeed)

    # wind = data.NO_WIND
    # wind = data.FAVORABLE_WIND
    # wind = data.OPPOSING_WIND
    wind = data.SAMPLE_WIND

    # fp = data.SQUARE_FLIGHT_PLAN # Barycentric fails due to cocircularity
    fp = data.SAMPLE_FLIGHT_PLAN

    process(fp, wind, cost_fn, barycentric=False)

    plt.show()


if __name__ == '__main__':
    fmt = "[%(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    main()
