"""Little collection of Gunnar's python stuff. Use at own risk."""

__all__ = ["plot", "ocean", "misc", "signal", "io"]

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
__version__ = "0.2.0"

# workaround for when whatever is defined as the default backend is not around:
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib as mpl

    mpl.use("Agg")

from . import io, misc, ocean, plot, signal
