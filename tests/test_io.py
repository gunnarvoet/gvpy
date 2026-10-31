import numpy as np

import gvpy as gv


def test_mat2dataset_leaves_input_dict_alone():
    """Regression test: mat2dataset renamed keys in the caller's dict."""
    m = dict(time=np.array([738000.0, 738001.0, 738002.0]), u=np.arange(3.0))
    ds = gv.io.mat2dataset(m)
    assert set(m) == {"time", "u"}
    assert ds.u.size == 3


def test_ants_reader_blank_line_and_to_xarray(tmp_path):
    """Regression test: blank lines raised IndexError and to_xarray passed self twice."""
    f = tmp_path / "test.vel"
    f.write_text(
        "#ANTS#PARAMS# lat{12.5} station{abc}\n"
        "#ANTS#FIELDS# {depth} {u}\n"
        "\n"
        "10 0.1\n"
        "20 0.2\n"
    )
    a = gv.io.ANTS(f)
    ds = a.to_xarray()
    np.testing.assert_array_equal(ds.depth, [10.0, 20.0])
    np.testing.assert_array_equal(ds.u, [0.1, 0.2])
    assert ds.lat.item() == 12.5
