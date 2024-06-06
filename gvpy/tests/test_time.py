import subprocess
import datetime
import numpy as np
import pytest

import gvpy


def test_datetime64_to_unix_time():
    """Test triangulation. Using data from BLT MP1."""

    def command_line_unix_time():
        result = subprocess.run(
            ["date", "-u", "+%s"], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        return int(result)

    unix_time = get_unixtime(np.datetime64(datetime.datetime.utcnow()))
    assert unix_time == command_line_unix_time()
