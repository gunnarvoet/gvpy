#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.adcp with adcp functions

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_raw_adcp(adcp):
	"""
    Plot raw RDI adcp dataset.

    Parameters
    ----------
    adcp : xarray.Dataset
        Raw RDI ADCP data read using gvpy.io.read_raw_rdi()

    Returns
    -------
    var : dtype
        description
    """

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))