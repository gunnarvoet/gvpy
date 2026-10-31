import subprocess
import sys

import numpy as np

import gvpy as gv


def test_import_does_not_load_ipynbname():
    """ipynbname pulls in the jupyter stack, which warns on import.

    Regression test: misc imported ipynbname at module level, and the jupyter_core
    deprecation warning broke test collection under filterwarnings=error.
    """
    code = "import sys, gvpy; print('ipynbname' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_near_scalar_target():
    A = np.arange(10.0)
    assert gv.misc.near(A, 3.2) == 3
    assert gv.misc.near(A, np.array([6.7])) == 7


def test_pretty_print_uses_indentstr(capsys):
    gv.misc.pretty_print(dict(a=dict(b=1)), indentstr="\t", print_values=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["a", "\tb"]
