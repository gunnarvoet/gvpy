import subprocess
import datetime
import numpy as np
import pytest

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
