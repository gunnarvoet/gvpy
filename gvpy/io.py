#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.io with in/out functions
"""

import datetime as dt
import re
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import scipy.io as spio
import xarray as xr
from munch import munchify
from seabird.cnv import fCNV

import gvpy


def loadmat(filename, onevar=False, verbose=False):
    """
    Load Matlab .mat files and return as dictionary with .dot-access.

    Parameters
    ----------
    filename : str
        Path to .mat file
    onevar : bool
        Set to true if there is only one variable in the mat file.

    Returns
    -------
    out : dict (Munch)
        Data in a munchified dictionary.
    """

    def _check_keys(dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            ni = np.size(dict[key])
            if ni <= 1:
                if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                    dict[key] = _todict(dict[key])
            else:
                for i in range(0, ni):
                    if isinstance(dict[key][i], spio.matlab.mio5_params.mat_struct):
                        dict[key][i] = _todict(dict[key][i])
        return dict

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = _check_keys(data)

    # Check if there is only one variable in the dataset. If so, directly
    # return only this variable as munchified dataset.
    if not onevar:
        dk = list(out.keys())
        actual_keys = [k for k in dk if k[:2] != "__"]
        if len(actual_keys) == 1:
            if verbose:
                print("found only one variable, returning munchified data structure")
            return munchify(out[actual_keys[0]])
        else:
            out2 = {}
            for k in actual_keys:
                out2[k] = out[k]
            return munchify(out2)

    # for legacy, keep the option in here as well.
    if onevar:
        # let's check if there is only one variable in there and return it
        kk = list(out.keys())
        outvars = []
        for k in kk:
            if k[:2] != "__":
                outvars.append(k)
        if len(outvars) == 1:
            if verbose:
                print("returning munchified data structure")
            return munchify(out[outvars[0]])
        else:
            if verbose:
                print("found more than one var...")
            return out
    else:
        return out


def savemat(out, filename):
    """Save dictionary to Matlab .mat file.

    Parameters
    ----------
    out : dict
        Dictionary with data to save. Variables in the .mat file will have dict keys as names.
    filename : str
        Path to .mat file
    """
    spio.savemat(filename, out, format="5")


def mtlb2datetime(matlab_datenum, strip_microseconds=False, strip_seconds=False):
    """
    Convert Matlab datenum format to python datetime.

    This version also works for vector input and strips
    milliseconds if desired.

    Parameters
    ----------
    matlab_datenum : float or np.array
        Matlab time vector.
    strip_microseconds : bool
        Get rid of microseconds (optional)
    strip_seconds : bool
        Get rid of seconds (optional)

    Returns
    -------
    t : np.datetime64
        Time in numpy's datetime64 format.
    """

    if np.size(matlab_datenum) == 1:
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
        t1 = day + dayfrac
        if strip_microseconds and strip_seconds:
            t1 = dt.datetime.replace(t1, microsecond=0, second=0)
        elif strip_microseconds:
            t1 = dt.datetime.replace(t1, microsecond=0)

    else:
        t1 = np.ones_like(matlab_datenum) * np.nan
        t1 = t1.tolist()
        nonan = np.isfinite(matlab_datenum)
        md = matlab_datenum[nonan]
        day = [dt.datetime.fromordinal(int(tval)) for tval in md]
        dayfrac = [dt.timedelta(days=tval % 1) - dt.timedelta(days=366) for tval in md]
        tt = [day1 + dayfrac1 for day1, dayfrac1 in zip(day, dayfrac)]
        if strip_microseconds and strip_seconds:
            tt = [dt.datetime.replace(tval, microsecond=0, second=0) for tval in tt]
        elif strip_microseconds:
            tt = [dt.datetime.replace(tval, microsecond=0) for tval in tt]
        tt = [np.datetime64(ti) for ti in tt]
        xi = np.where(nonan)[0]
        for i, ii in enumerate(xi):
            t1[ii] = tt[i]
        xi = np.where(~nonan)[0]
        for i in xi:
            t1[i] = np.datetime64("nat")
        t1 = np.array(t1)

    return t1


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
    """
    Read shipboard ADCP data file as produced by UHDAS.

    Parameters
    ----------
    ncfile : netcdf
        sADCP data from ship server.

    Returns
    -------
    xr.Dataset
        sADCP data in Dataset format.
    """
    sadcp = xr.open_dataset(ncfile)

    mdepth = sadcp.depth.median(dim="time")
    sadcp = sadcp.drop("depth")
    # sadcp['depth'] = (['depth_cell'], mdepth)
    sadcp = sadcp.rename_dims({"depth_cell": "z"})
    sadcp.coords["z"] = (["z"], mdepth.data)
    # Fix some attributes
    sadcp.z.attrs = dict(long_name="depth", units="m")
    sadcp.u.attrs = dict(long_name="u", units="m/s")
    sadcp.v.attrs = dict(long_name="v", units="m")
    sadcp.time.attrs = dict(long_name="time", units="")
    # Transpose (re-order dimensions)
    sadcp = sadcp.transpose("z", "time")

    return sadcp


def mat2dataset(m1):
    """
    Convert dictionary with data into xarray.Dataset

    Parameters
    ----------
    m1 : dict
        Dictionary with data, as ouput by gvpy.io.loadmat()

    Returns
    -------
    xr.Dataset
        Dataset with named variables
    """
    if "DateNum" in m1.keys() and "datenum" not in m1.keys():
        m1["datenum"] = m1.pop("DateNum")
    if "mtime" in m1.keys() and "datenum" not in m1.keys():
        m1["datenum"] = m1.pop("mtime")
    if "dtnum" in m1.keys() and "datenum" not in m1.keys():
        m1["datenum"] = m1.pop("dtnum")
    if "time" in m1.keys() and "datenum" not in m1.keys():
        m1["datenum"] = m1.pop("time")

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
    elif "P" in k:
        jj = m1["P"].shape[0]
    else:
        jj = 0

    if "lon" in k and type(m1["lon"]) != float:
        if len(m1["lon"].shape) == 1:
            ii = m1["lon"].shape[0]
    elif "dnum" in k:
        if len(m1["dnum"].shape) == 1:
            ii = m1["dnum"].shape[0]
    elif "datenum" in k:
        if len(m1["datenum"].shape) == 1:
            ii = m1["datenum"].shape[0]

    # maybe this is just a time series and not 2D...
    if not vars2d:
        is2d = False
    elif jj > 0 and jj == ii:
        is2d = False
    else:
        is2d = True

    if is2d:
        out = xr.Dataset(data_vars={"dummy": (["z", "x"], np.ones((jj, ii)) * np.nan)})
    else:
        out = xr.Dataset(data_vars={"dummy": (["x"], np.ones(ii) * np.nan)})

    # Assign 1d variables
    for v in vars1d:
        if m1[v].shape[0] == ii:
            out[v] = (["x"], m1[v])
        elif m1[v].shape[0] == jj:
            out[v] = (["z"], m1[v])

    # convert the usual suspects into coordinates
    if is2d:
        suspects = ["lon", "lat", "p", "z", "depth", "dep", "P"]
        for si in suspects:
            if si in vars1d:
                out.coords[si] = out[si]

    # convert time if possible
    for si in ["datenum", "dtnum", "dnum", "time"]:
        if si in vars1d and 1e8 > np.nanmedian(m1[si]) > 1e5:
            out.coords["time"] = (["x"], gvpy.time.mattime_to_datetime64(m1[si]))
    # we have a problem if there is a variable called 'time' in 2D.
    if "time" in vars2d:
        m1["time2d"] = m1.pop("time")
        vars2d = ["time2d" if v == "time" else v for v in vars2d]

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

    # remove entries without time stamp
    if "time" in out.coords:
        out = out.where(~np.isnat(out.time), drop=True)
    return out


def str_to_datetime64(timestr):
    """
    Convert date/time in str format to numpy's datetime64 format.

    Makes intermediate use of pandas datetime format, their string
    conversion seems to be much more capable than numpy's.

    Parameters
    ----------
    timestr : str
        Date/time

    Returns
    -------
    time : np.datetime64
        Time in numpy datetime64 format
    """
    return pd.to_datetime(timestr).to_datetime64()


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
    base = dt.datetime(baseyear, 1, 1, 0, 0, 0)
    time = [base + dt.timedelta(days=ti) for ti in yday - 1]
    # convert to numpy datetime64
    time64 = np.array([np.datetime64(ti, "ms") for ti in time])
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
    base = dt.datetime(baseyear, 1, 1, 0, 0, 0)
    time = [base + dt.timedelta(days=ti) for ti in yday]
    # convert to numpy datetime64
    time64 = np.array([np.datetime64(ti, "ms") for ti in time])
    return time64


def mpmat_load(file):
    """Read moored profiler data in .mat format processed with the MP
    processing toolbox in Matlab.

    Parameters
    ----------
    file : Path or str
        Path to MP dataset.

    Returns
    -------
    mp : xr.Dataset
        MP dataset.
    """
    tmp = loadmat(file)
    mp = mat2dataset(tmp)
    return mp


def mpmat_load_raw(path_raw_mat, n):
    path_raw_mat = _ensure_Path(path_raw_mat)
    file = path_raw_mat.joinpath(f"raw{n:04d}.mat")
    mpts = loadmat(file)
    mpts_time = mtlb2datetime(mpts.engtime)

    tmp = mpts["psdate"] + " " + mpts["pstart"]
    start_time = str_to_datetime64(tmp)
    tmp = mpts["pedate"] + " " + mpts["pstop"]
    stop_time = str_to_datetime64(tmp)

    mp = xr.Dataset(
        dict(epres=(["etime"], mpts.epres)),
        coords=dict(
            etime=(["etime"], mpts_time),
            esnum=(["etime"], mpts.esnum),
            csnum=(["csnum"], mpts.csnum),
        ),
    )
    evars = [
        "ecurr",
        "evolt",
        "engtime",
        "edpdt",
    ]
    for var in evars:
        mp[var] = (["etime"], mpts[var])

    timevars = [
        "psdate",
        "pedate",
        "pstart",
        "pstop",
    ]
    for var in timevars:
        mp.attrs[var] = mpts[var]
    mp.attrs["start"] = start_time
    mp.attrs["stop"] = stop_time

    cvars = ["csnum", "ccond", "ctemp", "cpres"]
    for var in cvars:
        mp[var] = (["csnum"], mpts[var])
    mp = ctd_time(mp)

    avars = [
        "Vab",
        "Vcd",
        "Vef",
        "Vgh",
        "aHx",
        "aHy",
        "aHz",
        "aTx",
        "aTy",
    ]
    mp.coords["asnum"] = (["asnum"], mpts["asnum"])
    for var in avars:
        mp[var] = (["asnum"], mpts[var])

    mp["Vx"] = (["asnum"], -(mp.Vab + mp.Vef) / (2 * 0.707))
    mp["Vy"] = (["asnum"], (mp.Vab - mp.Vef) / (2 * 0.707))
    mp["Vz1"] = (["asnum"], mp.Vx - mp.Vgh / 0.707)
    mp["Vz2"] = (["asnum"], -mp.Vx + mp.Vcd / 0.707)
    return mp


class ANTS(object):
    """
    Reader for ANTS data files.

    These may be .vel, .bt or .sh files as
    generated by Andreas Thurnherr's various toolboxes.
    Borrows from Andreas' matlab utilities.

    Parameters
    ----------
    filename : str or Path
        File to be read
    """

    def __init__(self, filename):
        self.filename = filename
        if not isinstance(self.filename, Path):
            self.filename = Path(self.filename)

        with open(self.filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        p_error = re.compile("^#ANTS#ERROR#")
        p_param = re.compile("^#ANTS#PARAMS#")
        p_values = re.compile(r"([\w\.]+){([^}]*)}")
        p_check_fields = re.compile("^#ANTS#FIELDS#")
        p_fields = re.compile("{([^}]*)}")
        p_data = re.compile("([^ \t]+)")

        test = []
        for c in content:
            if c[0] == "#":
                if p_error.match(c):
                    print("error")
                elif p_param.match(c):
                    tmp = p_values.findall(c)
                    for tmpi in tmp:
                        parameter, value = tmpi
                        if _is_number(value):
                            value = float(value)
                        setattr(self, parameter.replace(".", "_"), value)
                elif p_check_fields.match(c):
                    # we'll overwrite any previous results from lines like this
                    fieldnames = []
                    tmp = p_fields.findall(c)
                    for tmpi in tmp:
                        fieldnames.append(tmpi.replace(".", "_"))
            elif _is_number(c[:2]):
                tmp = p_data.findall(c)  # list of numbers as strings
                tmp = np.array([float(i) for i in tmp])  # convert to floats
                test.append(tmp)  # add to temporary list
        all_data = np.vstack(test)
        for i, f in enumerate(fieldnames):
            setattr(self, f, all_data[:, i])

    def to_xarray(self):
        """
        Convert ANTS object to xarray.Dataset.

        Returns
        -------
        ds : xarray.Dataset
            xarray data structure
        """
        return self._to_xarray(self)

    def _to_xarray(self):
        """
        Convert ANTS object to xarray.Dataset.

        Returns
        -------
        ds : xarray.Dataset
            xarray data structure
        """
        all_attrs = dir(self)
        cleaned_attrs = [x for x in all_attrs if x[0] != "_"]
        cdic = {ci: [] for ci in cleaned_attrs}
        ds = xr.Dataset()

        ck = list(cdic.keys())
        for ci in ck:
            a = getattr(self, ci)
            if isinstance(a, np.ndarray):
                cdic.pop(ci)
                ds[ci] = (["n"], a)

        ck = list(cdic.keys())
        for ci in ck:
            a = getattr(self, ci)
            if isinstance(a, float):
                cdic.pop(ci)
                ds[ci] = (["cast"], [a])

        ck = list(cdic.keys())
        for ci in ck:
            a = getattr(self, ci)
            if isinstance(a, str):
                cdic.pop(ci)
                ds.attrs[ci] = a

        # add source file name as attribute to dataset
        ds.attrs["filename"] = self.filename

        # change dim, coord based on file suffix
        if self.filename.suffix == ".VKE":
            ds = ds.swap_dims({"n": "depth"})
            ds.coords["depth"] = ds.depth
        if self.filename.suffix == ".wprof":
            ds = ds.swap_dims({"n": "depth"})
            for c in ["depth", "hab", "dc_depth", "uc_depth"]:
                if c in ds:
                    ds.coords[c] = ds[c]

        return ds


def results_to_latex(res, file):
    """Write dictionary with results to a latex file.

    In your latex document, use \\include{FileName} to read the document
    and then call variables as for e.g. \\OverallResult. Note that the file needs
    to be in the same directory as the main tex document.

    Parameters
    ----------
    res : dict
        Dictionary with results as values and new latex variables as keys.
    file : Path object
        Path to latex file to be generated.
    """

    def newcommand(name, val):
        """Format name and val into latex command"""
        fmt = "\\newcommand{{\\{name}}}[0]{{{action}}}"
        cmd = fmt.format(name=name, action=val)
        print(cmd)
        return cmd + "\n"

    cmds = []
    for key, values in res.items():
        cmds += newcommand(key, values)

    with open(file, "a") as fh:
        for cmd in cmds:
            fh.write(cmd)


def _is_number(s):
    """
    Check if string can be converted to a float.

    Parameters
    ----------
    s : str
        string

    Returns
    -------
    out : bool
        True if string can be converted to float, else False.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def _ensure_Path(path):
    path = Path(path) if not isinstance(path, Path) else path
    return path
