#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.misc with in/out functions

"""

from __future__ import print_function, division
import numpy as np
import xarray as xr
from gvpy.gvimport import mtlb2datetime
from seabird.cnv import fCNV
from datetime import datetime, timedelta
import gsw
import pandas as pd
from pycurrents.adcp.rdiraw import Multiread


def read_sbe_cnv(file, lat=0, lon=0):
    """
    Read Seabird SBE37 .cnv file and return as xarray.Dataset.

    Parameters
    ----------
    file : str
        Complete path to .cnv file
    lat : float
        Latitude (used for gsw calculations). Defaults to zero.
    lon : float
        Longitude (used for gsw calculations). Defaults to zero.

    Returns
    -------
    mc : xarray.Dataset
        Microcat data as Dataset with some metadata in the attributes.
    """
    # Read cnv file using Seabird package
    cnv = fCNV(file)

    # parse time
    mcyday = cnv["timeJV2"]
    start_time_str_all = cnv.attributes["start_time"]
    start_time_str = start_time_str_all.split("[")[0]
    base_year = pd.to_datetime(start_time_str).year
    mctime = yday1_to_datetime64(base_year, mcyday)
    # let's make sure the first time stamp we generated matches the string in the cnv file
    assert pd.to_datetime(np.datetime64(mctime[0], "s")) == pd.to_datetime(
        start_time_str
    )

    # data vars
    dvars = {"prdM": "p", "tv290C": "t"}
    mcdata = {}
    for k, di in dvars.items():
        if k in cnv.keys():
            # print(di, ':', k)
            mcdata[di] = (["time"], cnv[k])
    mc = xr.Dataset(data_vars=mcdata, coords={"time": mctime})
    mc.attrs["file"] = cnv.attributes["filename"]
    mc.attrs["sbe_model"] = cnv.attributes["sbe_model"]
    # conductivity
    cvars = {"cond0mS/cm": "c", "cond0S/m": "c"}
    for k, di in cvars.items():
        if k in cnv.keys():
            # convert from S/m to mS/cm as this is needed for gsw.SP_from_C
            if k == "cond0S/m":
                conductivity = cnv[k] * 10
            else:
                conductivity = cnv[k]
            mc[di] = (["time"], conductivity)

    # calculate oceanographic variables
    mc["SP"] = (["time"], gsw.SP_from_C(mc.c, mc.t, mc.p))
    if lat == 0 and lon == 0:
        print(
            "warning: absolute salinity, conservative temperature\n",
            "and density calculation may be inaccurate\n",
            "due to missing latitude/longitude",
        )
    mc["SA"] = (["time"], gsw.SA_from_SP(mc.SP, mc.p, lat=lat, lon=lon))
    mc["CT"] = (["time"], gsw.CT_from_t(mc.SA, mc.t, mc.p))
    mc["sg0"] = (["time"], gsw.sigma0(mc.SA, mc.CT))

    # add attributes
    attributes = {
        "p": dict(long_name="pressure", units="dbar"),
        "t": dict(long_name="in-situ temperature", units="°C"),
        "CT": dict(long_name="conservative temperature", units="°C"),
        "SA": dict(long_name="absolute salinity", units=r"kg/m$^3$"),
        "c": dict(long_name="conductivity", units="mS/cm"),
        "SP": dict(long_name="practical salinity", units=""),
        "sg0": dict(long_name=r"potential density $\sigma_0$", units=r"kg/m$^3$"),
    }
    for k, att in attributes.items():
        if k in list(mc.variables.keys()):
            mc[k].attrs = att

    return mc


def read_sadcp(ncfile):
    sadcp = xr.open_dataset(ncfile)

    mdepth = sadcp.depth.median(dim="time")
    sadcp = sadcp.drop("depth")
    # sadcp['depth'] = (['depth_cell'], mdepth)
    sadcp = sadcp.rename_dims({"depth_cell": "z"})
    sadcp.coords["z"] = (["z"], mdepth)
    # Fix some attributes
    sadcp.z.attrs = dict(long_name="depth", units="m")
    sadcp.u.attrs = dict(long_name="u", units="m/s")
    sadcp.v.attrs = dict(long_name="v", units="m")
    sadcp.time.attrs = dict(long_name="time", units="")
    # Transpose (re-order dimensions)
    sadcp = sadcp.transpose("z", "time")

    return sadcp


def mat2dataset(m1):
    k = m1.keys()

    varsint = []
    vars1d = []
    vars2d = []

    for ki in k:
        try:
            tmp = m1[ki].shape
            if len(tmp) == 1:
                vars1d.append(ki)
            elif len(tmp) == 2:
                vars2d.append(ki)
        except:
            tmp = None
            varsint.append(ki)
    # let's find probable dimensions. usually we have p or z for depth
    if "z" in k:
        jj = m1["z"].shape[0]
    elif "p" in k:
        jj = m1["p"].shape[0]

    if "lon" in k:
        if len(m1["lon"].shape) == 1:
            ii = m1["lon"].shape[0]
    elif "dnum" in k:
        if len(m1["dnum"].shape) == 1:
            ii = m1["dnum"].shape[0]

    out = xr.Dataset(data_vars={"dummy": (["z", "x"], np.ones((jj, ii)) * np.nan)})
    # get 1d variables
    for v in vars1d:
        if m1[v].shape[0] == ii:
            out[v] = (["x"], m1[v])
        elif m1[v].shape[0] == jj:
            out[v] = (["z"], m1[v])

    # convert the usual suspects into variables
    suspects = ["lon", "lat", "p", "z", "depth", "dep"]
    for si in suspects:
        if si in vars1d:
            out.coords[si] = out[si]

    # convert time if possible
    for si in ["datenum", "dtnum", "dnum"]:
        if si in vars1d and np.median(m1[si]) > 1e5:
            out.coords["time"] = (["x"], mtlb2datetime(m1[si]))

    # get 2d variables
    for v in vars2d:
        if m1[v].shape[0] == ii and m1[v].shape[1] == jj:
            out[v] = (["z", "x"], m1[v])
        elif m1[v].shape[0] == jj and m1[v].shape[1] == ii:
            out[v] = (["z", "x"], m1[v])

    # swap dim x for time if we have a time vector
    if "time" in out.coords:
        out = out.swap_dims({"x": "time"})

    # drop dummy
    out = out.drop(["dummy"])
    return out


def yday1_to_datetime64(baseyear, yday):
    """
    Convert year day (starting at yday 1) to numpy's datetime64 format.

    Parameters
    ----------
    baseyear : int
        Base year
    yday : float
        Year day

    Returns
    -------
    time : np.datetime64
        Time in numpy datetime64 format
    """
    base = datetime(baseyear, 1, 1, 0, 0, 0)
    time = [base + timedelta(days=ti) for ti in yday - 1]
    # convert to numpy datetime64
    time64 = [np.datetime64(ti, "ms") for ti in time]
    return time64


def yday0_to_datetime64(baseyear, yday):
    """
    Convert year day (starting at yday 0) to numpy's datetime64 format.

    Parameters
    ----------
    baseyear : int
        Base year
    yday : float
        Year day

    Returns
    -------
    time : np.datetime64
        Time in numpy datetime64 format
    """
    base = datetime(baseyear, 1, 1, 0, 0, 0)
    time = [base + timedelta(days=ti) for ti in yday]
    # convert to numpy datetime64
    time64 = [np.datetime64(ti, "ms") for ti in time]
    return time64


def read_raw_rdi(file, auxillary_only=False):
    """
    Read raw RDI ADCP data file and return as xarray Dataset.

    Parameters
    ----------
    file : str or Path
        Path to raw data file.
    auxillary_only : bool
        Set to True to ignore 2d fields. (default False)

    Returns
    -------
    rdi : xarray.Dataset
        Raw ADCP data in xarray Dataset format.
    """

    m = Multiread(file, "wh")

    if auxillary_only:
        radcp = m.read(varlist=["FixedLeader"])
    else:
        if "BottomTrack" in m.available_varnames:
            radcp = m.read(
                varlist=[
                    "Velocity",
                    "PercentGood",
                    "Intensity",
                    "Correlation",
                    "BottomTrack",
                ]
            )
        else:
            radcp = m.read(varlist=["Velocity", "PercentGood", "Intensity", "Correlation"])
    # convert time
    adcptime = yday0_to_datetime64(radcp.yearbase, radcp.dday)

    jj = np.squeeze(radcp.dep.shape)
    assert radcp.nbins == jj
    ii = np.squeeze(radcp.dday.shape)
    assert radcp.nprofs == ii
    varsii = [
        "num_pings",
        "dday",
        "ens_num",
        "temperature",
        "heading",
        "pitch",
        "roll",
        "XducerDepth",
    ]
    vars2d = [
        "vel1",
        "vel2",
        "vel3",
        "vel4",
        "amp1",
        "amp2",
        "amp3",
        "amp4",
        "cor1",
        "cor2",
        "cor3",
        "cor4",
        "pg1",
        "pg2",
        "pg3",
        "pg4",
    ]

    out = xr.Dataset(data_vars={"dummy": (["z", "time"], np.ones((jj, ii)) * np.nan)})

    # get 1d variables
    for v in varsii:
        out[v] = (["time"], radcp[v])
    # add pressure
    out['Pressure'] = (["time"], radcp.VL['Pressure']/1000)

    # get 2d variables
    if auxillary_only is False:
        for v in vars2d:
            if radcp[v].shape[0] == ii and radcp[v].shape[1] == jj:
                out[v] = (["z", "time"], np.transpose(radcp[v]))
            elif radcp[v].shape[0] == jj and radcp[v].shape[1] == ii:
                out[v] = (["z", "time"], radcp[v])

    out.coords["time"] = (["time"], adcptime)
    out.coords["z"] = (["z"], radcp.dep)

    # bottom tracking
    if 'bt_vel' in radcp.keys():
        out['bt_vel'] = (["time", "beam"], radcp.bt_vel)
        out['bt_depth'] = (["time", "beam"], radcp.bt_depth)
        out.coords["beam"] = np.array([1, 2, 3, 4])

    # drop dummy
    out = out.drop(["dummy"])

    # set a few attributes
    out.attrs["sonar"] = radcp.sonar.sonar
    out.attrs["coordsystem"] = radcp.trans.coordsystem
    out.attrs["pingtype"] = radcp.pingtype

    return out
