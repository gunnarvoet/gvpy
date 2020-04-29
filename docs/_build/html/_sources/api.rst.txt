.. currentmodule:: gvpy

#############
API reference
#############

This page provides an auto-generated summary of gvpy's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

I/O
===
.. autosummary::
    :toctree: generated/

    io.loadmat
    io.mat2dataset
    io.read_sbe_cnv
    io.read_sadcp
    io.ANTS
    io.ANTS._to_xarray


Ocean
=====

Dataset Access
----------------
.. autosummary::
    :toctree: generated/

    ocean.woa_get_ts
    ocean.woce_climatology
    ocean.woce_argo_profile
    ocean.smith_sandwell
    ocean.smith_sandwell_section
    ocean.tpxo_extract

Calculations
----------------
.. autosummary::
    :toctree: generated/

    ocean.nsqfcn
    ocean.tzfcn
    ocean.bathy_section
    ocean.inertial_period
    ocean.inertial_frequency
    ocean.uv2speeddir


Signal Processing
=================
.. autosummary::
    :toctree: generated/

    signal.lowpassfilter
    signal.bandpassfilter


Plotting
========

Figures and Axes
----------------
.. autosummary::
    :toctree: generated/

    plot.axstyle
    plot.newfigyy
    plot.add_cax
    plot.ydecrease
    plot.ysym
    plot.xsym
    plot.xytickdist
    plot.concise_date
    plot.cartopy_axes

Colors
----------------
.. autosummary::
    :toctree: generated/

    plot.colcyc10
    misc.cmap_div

Plotting Methods
----------------
.. autosummary::
    :toctree: generated/

    plot.multi_line
    plot.pcm

Saving
----------------
.. autosummary::
    :toctree: generated/

    plot.png
    plot.figsave


Little Helpers
================
.. autosummary::
    :toctree: generated/

    ocean.lonlatstr
    io.mtlb2datetime
    io.str_to_datetime64
    io.yday1_to_datetime64
    io.yday0_to_datetime64
    misc.near
    misc.nearidx
    misc.nearidx2
    misc.getshape
    misc.qpload
    misc.qpsave

Debugging
=========
.. autosummary::
    :toctree: generated/

    misc.extract

