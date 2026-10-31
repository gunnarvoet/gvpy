import numpy as np
import xarray as xr

from gvpy import trilaterate


def test_trilateration():
    """Test triangulation. Using data from BLT MP1."""
    lat = np.array([54.24411189, 54.23283283, 54.23704613])
    lon = np.array([-11.953817, -11.95783703, -11.93493788])
    pos = [(loni, lati) for loni, lati in zip(lon, lat)]
    ranges = np.array([2146, 2204, 2221])

    plan_lon, plan_lat = -11 - 56.958 / 60, 54 + 14.334 / 60
    bottom_depth = 2034
    # flat synthetic bathymetry so the test does not need Smith & Sandwell
    topo = xr.DataArray(
        np.full((21, 21), -bottom_depth, dtype=float),
        coords=dict(
            lat=np.linspace(plan_lat - 0.1, plan_lat + 0.1, 21),
            lon=np.linspace(plan_lon - 0.2, plan_lon + 0.2, 21),
        ),
        dims=("lat", "lon"),
    )
    mp1 = trilaterate.Trilateration(
        "MP1",
        plan_lon=plan_lon,
        plan_lat=plan_lat,
        bottom_depth=bottom_depth,
        topo=topo,
    )
    for p, d in zip(pos, ranges):
        mp1.add_ranges(distances=d, pos=p)

    mp1.trilaterate(i=0)

    tmp = mp1.to_netcdf()
    assert tmp.offset < 100
    assert tmp.depth_actual == -bottom_depth


def test_trilateration_without_bathymetry(monkeypatch):
    monkeypatch.setattr(trilaterate.gv.ocean, "smith_sandwell", lambda **kwargs: None)
    t = trilaterate.Trilateration("MP1", plan_lon=-11.95, plan_lat=54.24)
    assert t.topo is None


def test_trilateration_result_repr_without_offset():
    r = trilaterate.TrilaterationResult(lon=1.0, lat=2.0, error=3.0)
    assert "offset=None" in repr(r)
