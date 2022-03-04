import numpy as np
import pytest

from gvpy import trilaterate


def test_trilateration():
    """Test triangulation. Using data from BLT MP1."""
    lat = np.array([54.24411189, 54.23283283, 54.23704613])
    lon = np.array([-11.953817, -11.95783703, -11.93493788])
    pos = [(loni, lati) for loni, lati in zip(lon, lat)]
    ranges = np.array([2146, 2204, 2221])

    plan_lon, plan_lat = -11-56.958/60, 54 + 14.334/60
    bottom_depth = 2034
    mp1 = trilaterate.Trilateration(
        "MP1",
        plan_lon=plan_lon,
        plan_lat=plan_lat,
        bottom_depth=bottom_depth,
    )
    for p, d in zip(pos, ranges):
        mp1.add_ranges(distances=d, pos=p)

    t = mp1.trilaterate(i=0)

    tmp = mp1.to_netcdf()
    assert tmp.offset < 100
