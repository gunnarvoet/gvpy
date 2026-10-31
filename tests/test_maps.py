import numpy as np
import xarray as xr

import gvpy as gv


def topo():
    return xr.DataArray(
        np.random.default_rng(0).normal(size=(20, 30)),
        coords=dict(lat=np.linspace(10, 11, 20), lon=np.linspace(-5, -3, 30)),
        dims=("lat", "lon"),
    )


def test_add_dist_coords_leaves_input_alone():
    ds = topo().to_dataset(name="z")
    out = gv.maps.add_dist_coords(ds)
    assert "xdist" not in ds.coords
    assert out.xdist.size == 30


def test_hillshade_extent_from_dataarray_coords():
    """Regression test: `"lon" in da` tests data values, so coords were never used."""
    hs = gv.maps.HillShade(topo())
    assert hs.topo_extent == (-5.0, -3.0, 11.0, 10.0)


def test_hillshade_extent_from_numpy_lon_lat():
    """Regression test: numpy lon/lat gave memoryview extents."""
    t = topo()
    hs = gv.maps.HillShade(t.data, lon=t.lon.data, lat=t.lat.data)
    assert hs.topo_extent == (-5.0, -3.0, 11.0, 10.0)
