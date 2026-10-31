import datetime
import subprocess

import numpy as np

import gvpy as gv


def test_datetime64_to_unix_time():
    """Test conversion from np.datetime64 to Unix time.

    Note
    ----
    We can compare against command line output of `date` as Unix time is in
    seconds; both commands will thus be run at the same time.
    """

    def command_line_unix_time():
        result = subprocess.run(
            ["date", "-u", "+%s"], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        return int(result)

    dt_now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    dt64_now = np.datetime64(dt_now)
    unix_time = gv.time.datetime64_to_unix_time(dt64_now)

    assert unix_time == command_line_unix_time()


def test_convert_units_second_resolution_array():
    """Regression test: datetime64 arrays not in ns or ms raised UnboundLocalError."""
    t = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="M8[s]")
    out = gv.time.convert_units(t, unit="ms")
    assert out.dtype == np.dtype("M8[ms]")
    np.testing.assert_array_equal(out, t)
