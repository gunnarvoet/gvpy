#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.misc with miscellaneous functions"""

import sys
import subprocess
import ctypes
import inspect
import warnings
from loguru import logger
import numpy as np
from IPython import get_ipython
import ipynbname


def hello():
    print("hello, world!")


def near(A, target):
    """
    Find index of value in A closest to target.
    A must be sorted!
    """
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    assert len(idx) == 1, "returning two values"
    return idx


def nearidx(array, value):
    """
    Find index of value in array closest to target.
    No need for array to be sorted.
    """
    n = [np.abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx


def nearidx2(array, value):
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx


def getshape(d):
    """
    Get dict with info on dict d
    """
    if isinstance(d, dict):
        return {k: np.shape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None


def cmap_div(
    numcolors=11,
    name="custom_div_cmap",
    mincol="blue",
    midcol="white",
    maxcol="red",
):
    """Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    Adapted from http://pyhogs.github.io/colormap-examples.html
    """

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        name=name, colors=[mincol, midcol, maxcol], N=numcolors
    )
    return cmap


def log():
    """Set up stdout logger using loguru.

    Returns
    -------
    logger : loguru.Logger
        Logger object. Use to log messages at different levels.

    Info
    ----
    For example:
    ```
    log = log()
    log.info("hey there")
    ```

    [loguru docs](https://loguru.readthedocs.io/en/stable/index.html)

    """
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<e>{time:YYYY-MM-DD HH:mm:ss}</e> | {level} | <level>{message}</level>",
    )
    return logger


def jupaexit():
    """
    jupaexit()
    Exit attached console without killing kernel
    """
    exit(keep_kernel=True)


def qpload(filename):
    """
    Quick pickle - load

    Parameters
    ----------
    filename : str
        File name including path

    Returns
    -------
    var : dtype
        loaded variables
    """
    import pickle

    f = open(filename, "rb")
    out = pickle.load(f)
    f.close()
    return out


def qpsave(filename, vars):
    """
    Quick pickle - save

    Parameters
    ----------
    filename : str
        File name including path

    vars : list
        List with names of variables to be saved
    """
    import pickle

    f = open(filename, "wb")
    pickle.dump(vars, f)
    f.close()


def extract(prepend="xx"):
    """Copies the variables of the caller up to iPython. Useful when in
    debugging mode.

    Parameters
    ----------
    prepend : str, optional
        String to prepend to each variable name. Defaults to 'xx'.

    By default returns variables with xx prepended to their names as not to clutter the
    workspace.

    Example
    -------
        def f():
            a = 'hello world'
            assert 1 == 0
            extract()

        f() # raises an error

        print(xxa) # prints 'hello world'

    Notes
    -----
    see https://andyljones.com/posts/post-mortem-plotting.html
    """

    frames = inspect.stack()
    caller = frames[1].frame
    _name, ls, gs = caller.f_code.co_name, caller.f_locals, caller.f_globals

    ipython = [f for f in inspect.stack() if f.filename.startswith("<ipython-input")][
        -1
    ].frame

    ipython.f_locals.update(
        {"{}{}".format(prepend, k): v for k, v in gs.items() if k[:2] != "__"}
    )
    ipython.f_locals.update(
        {"{}{}".format(prepend, k): v for k, v in ls.items() if k[:2] != "__"}
    )

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(ipython), ctypes.c_int(0))


def latex_float(f, decimals=0):
    """Print float as string formatted for use in latex.

    Parameters
    ----------
    f : float
        Float to be formatted.
    decimals : int, optional
        Number of decimals to show. Default 0.

    Returns
    -------
    str
    """
    fmt = "{{:1.{decimals}e}}".format(decimals=decimals)
    float_str = fmt.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def warnless(verbose=False):
    """Suppress common annoying warnings."""
    if verbose:
        print("Ignore the following warnings:")
    messages = ["Mean of empty slice", "invalid value encountered in greater"]
    for mi in messages:
        if verbose:
            print(mi)
        warnings.filterwarnings("ignore", message=mi)


def pretty_print(d, indent=0, indentstr="   ", print_values=True):
    r"""Pretty print a nested dictionary.

    Parameters
    ----------
    d : `dict`
        Dicitonary to print.
    indent : `int`, optional
        Extra indent (this times `indentstr`).
    indentstr : `str`, optional
        Indent str. Defaults to three spaces. Change to '\t' for tabs if
        desired.
    print_values : bool, optional
        Show dictionary values. Defaults to True.
    """
    indentstr = "   "
    for key, value in d.items():
        print(indentstr * indent + str(key))
        if isinstance(value, dict):
            pretty_print(
                value,
                indent + 1,
                indentstr=indentstr,
                print_values=print_values,
            )
        else:
            if print_values:
                print(indentstr * (indent + 1) + str(value))


def is_notebook():
    s = get_ipython().__class__.__name__
    if s == "ZMQInteractiveShell":
        return True
    else:
        return False


def is_ipython():
    s = get_ipython().__class__.__name__
    if s == "TerminalInteractiveShell":
        return True
    else:
        return False


def nb_name():
    return ipynbname.name()


def nb_path():
    return ipynbname.path()


def connect_to_server(server, drive):
    """Mount smb drive in macOS.

    Parameters
    ----------
    server : str
        Server address (without smb://)
    drive : str
        Drive to mount
    """
    command = 'mount volume "smb://{}/{}"'.format(server, drive)
    _output = subprocess.run(["osascript", "-e", command], capture_output=True)


def yes_or_no(prompt, default="y"):
    while True:
        response = input(prompt + " [Y/n]:").lower()
        if response == "y" or response == "n":
            return response == "y"
        elif response == "":
            return default == "y"
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def determine_header_length(file, comment_char="#"):
    header_length = 1
    with open(file) as f:
        g = f.readline()
        while g[0] == comment_char:
            header_length += 1
            g = f.readline()
    return header_length
