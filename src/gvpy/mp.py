#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.mp with functions for data collected with McLane Moored Profilers."""

from pathlib import Path
import gsw
import numpy as np
import scipy as sp
import xarray as xr
import mixsea as mx
import pandas as pd

import gvpy as gv
from . import io


def load_proc_mat(file, subvar=None):
    """Read moored profiler data in .mat format processed with the MP
    processing toolbox in Matlab.

    Parameters
    ----------
    file : Path or str
        Path to MP dataset.
    subvar : str
        Key for dictionary entry if MP data are contained in a subvariable
        within the .mat-file.

    Returns
    -------
    mp : xr.Dataset
        MP dataset.
    """
    tmp = io.loadmat(file)
    if subvar is not None:
        tmp = tmp[subvar]
    mp = io.mat2dataset(tmp)
    mp = mp.rename_dims(z="depth")
    mp = mp.rename_vars(z="depth")
    mp = mp.dropna(dim="depth", how="all")
    mp.coords["profile"] = mp.id
    mp = mp.drop("id")
    mp = add_hab(mp)
    lon = np.round(mp.lon.median().data, decimals=4)
    lat = np.round(mp.lat.median().data, decimals=4)
    mp = mp.drop(["lon", "lat"])
    mp.attrs["lon"] = lon
    mp.attrs["lat"] = lat

    if "time2d" in mp:
        time2d = np.array(
            [
                gv.time.mattime_to_datetime64(mpi.time2d.data.squeeze())
                for g, mpi in mp.groupby("time")
            ]
        )
        mp.coords["time2"] = (("depth", "time"), time2d.transpose())
        mp = mp.drop("time2d")

    atts = dict(
        depth=dict(long_name="depth", units="m"),
        p=dict(long_name="pressure", units="dbar"),
        th=dict(long_name=r"$\Theta$", units="°C"),
        t=dict(long_name="in-situ temperature", units="°C"),
        s=dict(long_name="salinity", units=""),
        c=dict(long_name="conductivity", units="mS/cm"),
        u=dict(long_name="u", units="m/s"),
        v=dict(long_name="v", units="m/s"),
        w=dict(long_name="w", units="m/s"),
        sgth=dict(long_name=r"$\sigma_\Theta$", units="kg/m$^3$"),
    )

    for k, v in atts.items():
        for ki, vi in v.items():
            mp[k].attrs[ki] = vi

    return mp


def load_raw_mat_file(file):
    """Read raw moored profiler time series converted to .mat format with the
    MP processing toolbox in Matlab.

    Parameters
    ----------
    file : Path or str
        Path to raw .mat file (usually under raw/ in the data directory).

    Returns
    -------
    mp : xr.Dataset
        MP dataset.
    """
    # path_raw_mat = io._ensure_Path(path_raw_mat)
    # file = path_raw_mat.joinpath(f"raw{n:04d}.mat")
    mpts = io.loadmat(file)
    mpts_time = io.mtlb2datetime(mpts.engtime)
    mpts_time = mpts_time.astype("<M8[ns]")

    tmp = mpts["psdate"] + " " + mpts["pstart"]
    start_time = io.str_to_datetime64(tmp)
    tmp = mpts["pedate"] + " " + mpts["pstop"]
    stop_time = io.str_to_datetime64(tmp)

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

    mp = acm_path_to_instrument_coordinate(mp)
    return mp


def load_raw_mat(path_raw_mat, n):
    """Read raw moored profiler time series for one deployment converted to
    .mat format with the MP processing toolbox in Matlab.

    Parameters
    ----------
    path_raw_mat : Path or str
        Path to raw .mat files (usually mat/ in the data directory).
    n : int or list[int] or range or 'all'
        Profile number or range or all.

    Returns
    -------
    mp : xr.Dataset or list[xr.Dataset]
        MP dataset or list thereof.
    """
    path_raw_mat = io._ensure_Path(path_raw_mat)
    if isinstance(n, int):
        file = path_raw_mat.joinpath(f"raw{n:04d}.mat")
        return load_raw_mat_file(file)
    elif isinstance(n, list) or isinstance(n, range):
        files = [path_raw_mat.joinpath(f"raw{ni:04d}.mat") for ni in n]
        mpa = [load_raw_mat_file(fi) for fi in files]
        return mpa
    elif isinstance(n, str) and n == "all":
        files = sorted(path_raw_mat.glob("raw*.mat"))
        mpa = [load_raw_mat_file(fi) for fi in files]
        return mpa


def concat_raw(rawlist):
    ctd = []
    acm = []
    eng = []
    for mp in rawlist:
        c, a, e = separate_raw(mp)
        ctd.append(c)
        acm.append(a)
        eng.append(e)
    ctd = xr.concat(ctd, dim="ctime")
    eng = xr.concat(eng, dim="etime")
    acm = xr.concat(acm, dim="n")
    return ctd, acm, eng


def separate_raw(mpraw):
    ctdvars = ["cpres", "ctemp", "ccond", "csnum", "ctime"]
    ctd = mpraw[ctdvars]
    engvars = ["epres", "ecurr", "evolt", "edpdt", "esnum", "engtime", "etime"]
    eng = mpraw[engvars]
    acm = mpraw.drop(ctdvars).drop(engvars)
    return ctd, acm, eng


def read_eng(file):
    """Read MP engineering file.

    There are several ways to unpack the raw data files
    (with and without header, comma-separated or space-separated).
    This routine tries to take care of all of them."""

    def test_for_header(file):
        with open(file, "r") as f:
            count = 1
            a = f.readline().strip()
            if a == "":
                a = f.readline().strip().strip("\n")
                count += 1
            if a.startswith("Profile"):
                has_header = True
                while a.startswith("Date") is False:
                    a = f.readline().strip().strip("\n")
                    count += 1
            else:
                has_header = False
        return has_header, count

    def test_for_footer(file):
        with open(file, "r") as f:
            lines = f.readlines()
            fcount = 1
            a = lines[-fcount]
            if len(a) > 2 and a.strip()[2] == "/":
                has_footer = False
            else:
                has_footer = True
                searching_last_row = True
                while searching_last_row:
                    fcount += 1
                    a = lines[-fcount]
                    if len(a) > 2 and a.strip()[2] == "/":
                        searching_last_row = False
                        fcount -= 1
        return has_footer, fcount

    def determine_delimiter(file):
        with open(file, "r") as f:
            a = f.readline().strip()
            if a == "":
                a = f.readline().strip().strip("\n")
            if a.startswith("Profile"):
                has_header = True
                while a.startswith("Date") is False:
                    a = f.readline().strip().strip("\n")
            else:
                has_header = False
            if "," in a:
                has_comma = True
            else:
                has_comma = False
        if has_comma:
            return ","
        else:
            return "\\s+"

    has_header, count = test_for_header(file)
    has_footer, fcount = test_for_footer(file)
    skiprows = count if has_header else 0
    skipfooter = fcount if has_footer else 0
    sep = determine_delimiter(file)
    cols = ["datestr", "timestr", "current", "voltage", "pressure"]
    if sep == ",":
        cols = ["timestr", "current", "voltage", "pressure"]
    df = pd.read_csv(
        file,
        names=cols,
        skiprows=skiprows,
        skipfooter=skipfooter,
        engine="python",
        sep=sep,
    )
    if sep == ",":
        df["time"] = pd.to_datetime(df["timestr"])
    else:
        df["time"] = pd.to_datetime(df["datestr"] + " " + df["timestr"])
    ds = df.to_xarray()
    ds = ds.swap_dims(index="time")
    ds.current.attrs = dict(long_name="current", units="mA")
    ds.voltage.attrs = dict(long_name="voltage", units="V")
    ds.pressure.attrs = dict(long_name="pressure", units="dbar")
    return ds


def read_ctd(file):
    def test_for_header(file):
        with open(file, "r") as f:
            count = 1
            a = f.readline().strip()
            while a == "":
                a = f.readline().strip().strip("\n")
                count += 1
            if a.startswith("Profile"):
                has_header = True
                while "." not in a and count < 20:
                    a = f.readline().strip().strip("\n")
                    count += 1
            else:
                has_header = False
        return has_header, count

    def test_for_footer(file):
        with open(file, "r") as f:
            lines = f.readlines()
            fcount = 1
            a = lines[-fcount]
            if len(a) > 2 and "." in a:
                has_footer = False
            else:
                has_footer = True
                searching_last_row = True
                while searching_last_row:
                    fcount += 1
                    a = lines[-fcount]
                    if len(a) > 2 and "." in a:
                        searching_last_row = False
                        fcount -= 1
        return has_footer, fcount

    def determine_delimiter(file):
        with open(file, "r") as f:
            count = 1
            a = f.readline().strip()
            while "." not in a and count < 20:
                a = f.readline().strip().strip("\n")
                count += 1
            if "," in a:
                has_comma = True
            else:
                has_comma = False
        if has_comma:
            return ","
        else:
            return "\\s+"

    has_header, count = test_for_header(file)
    has_footer, fcount = test_for_footer(file)
    skiprows = count if has_header else 0
    skipfooter = fcount if has_footer else 0
    sep = determine_delimiter(file)
    df = pd.read_csv(
        file,
        names=["c", "t", "p", "freq"],
        skiprows=skiprows,
        skipfooter=skipfooter,
        engine="python",
        sep=sep,
    )
    ds = df.to_xarray()
    ds.c.attrs = dict(long_name="conductivity", units="mS/cm")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.p.attrs = dict(long_name="pressure", units="dbar")
    ds.freq.attrs = dict(long_name="frequency", units="Hz")
    return ds


def read_acm(file):
    """Read FSI ACM file as generated by McLane unpacker software.
    Should work with/without header and with either comma or space as separator.

    Parameters
    ----------
    file : str or Path

    Returns
    -------
    xr.Dataset
    """

    def test_for_header(file):
        with open(file, "r") as f:
            count = 1
            a = f.readline().strip()
            while a == "":
                a = f.readline().strip().strip("\n")
                count += 1
            if a.startswith("Profile"):
                has_header = True
                while "." not in a and count < 20:
                    a = f.readline().strip().strip("\n")
                    count += 1
            else:
                has_header = False
        return has_header, count

    def test_for_footer(file):
        with open(file, "r") as f:
            lines = f.readlines()
            fcount = 1
            a = lines[-fcount]
            if len(a) > 2 and "." in a:
                has_footer = False
            else:
                has_footer = True
                searching_last_row = True
                while searching_last_row:
                    fcount += 1
                    a = lines[-fcount]
                    if len(a) > 2 and "." in a:
                        searching_last_row = False
                        fcount -= 1
        return has_footer, fcount

    def determine_delimiter(file):
        with open(file, "r") as f:
            count = 1
            a = f.readline().strip()
            while "." not in a and count < 20:
                a = f.readline().strip().strip("\n")
                count += 1
            if "," in a:
                has_comma = True
            else:
                has_comma = False
        if has_comma:
            return ","
        else:
            return "\\s+"

    has_header, count = test_for_header(file)
    has_footer, fcount = test_for_footer(file)
    skiprows = count if has_header else 0
    skipfooter = fcount if has_footer else 0
    sep = determine_delimiter(file)
    df = pd.read_csv(
        file,
        names=["aTx", "aTy", "aHx", "aHy", "aHz", "Vab", "Vcd", "Vef", "Vgh"],
        skiprows=skiprows,
        skipfooter=skipfooter,
        engine="python",
        sep=sep,
    )
    ds = df.to_xarray()
    ds = ds.rename_dims(index="asnum")
    ds = ds.rename_vars(index="asnum")
    ds = acm_path_to_instrument_coordinate(ds)
    # ds.c.attrs = dict(long_name='conductivity', units='mS/cm')
    # ds.t.attrs = dict(long_name='temperature', units='°C')
    # ds.p.attrs = dict(long_name='pressure', units='dbar')
    # ds.freq.attrs = dict(long_name='frequency', units='Hz')
    return ds


def read_compass_cal_acm(file: str | Path) -> np.ndarray:
    """Read FSI ACM output as logged from the terminal during compass calibration.

    The function ignores lines not containing exactly five values (Tx, Ty, Hx,
    Hy, Hz) and ignores any trailing '...'

    Parameters
    ----------
    file : str or Path

    Returns
    -------
    np.ndarray
    """
    # Initialize a list to hold the parsed rows of data.
    parsed_rows = []

    with open(file, "r") as f:
        for line in f:
            # Strip any leading/trailing whitespace from the line.
            line = line.strip()

            # Ignore lines that do not have a recognizable number.
            if not any(char.isdigit() or char in "+-." for char in line):
                continue

            # Remove the trailing '...' if it exists.
            if line.endswith("..."):
                line = line[:-3].strip()

            # Use a nested try-except block to handle lines that might not be
            # valid numeric data (e.g., header rows).
            try:
                # Split the line by whitespace and convert each part to a float.
                numbers = [float(x) for x in line.split()]

                # Check if the line contains exactly five numbers.
                if len(numbers) == 5:
                    # If the line is valid, add the list of numbers to our rows.
                    parsed_rows.append(numbers)

            except ValueError:
                # If a ValueError occurs, it means the line contained non-numeric
                # data, so we simply ignore it and move to the next line.
                continue

    # Convert the list of parsed rows into a NumPy array and return it.
    return np.array(parsed_rows)


def ctd_time(mpraw):
    """Align ctd pressure and engineering pressure to create ctd time vector.

    Parameters
    ----------
    mpraw : xr.Dataset
        Raw MP dataset.

    Returns
    -------
    mpraw : xr.Dataset
        Raw MP dataset.

    Notes
    -----
    From MP_mag_pgrid.m with a slight change as only the pressure time series
    for the times when the profiler is moving in the vertical are compared.
    """
    mask = (mpraw.epres != 0) & (np.absolute(mpraw.edpdt) > 10)
    mpcut = mpraw.where(mask, drop=True)

    nep = 0
    mep = -1

    diffpn = np.absolute(mpraw.cpres - mpcut.epres.isel(etime=nep))
    ncp = diffpn.argmin()

    diffpm = np.absolute(mpraw.cpres - mpcut.epres.isel(etime=mep))
    mcp = diffpm.argmin()

    ctime = np.arange(len(mpraw.cpres))
    ctime = ctime - ctime[ncp]
    ctdsamplerate = (mpcut.etime[mep] - mpcut.etime[nep]) / (mcp - ncp)
    ctime = mpcut.etime[nep].data + ctime * ctdsamplerate.data
    mpraw.coords["ctime"] = (["csnum"], ctime)
    mpraw = mpraw.swap_dims({"csnum": "ctime"})
    return mpraw


def acm_path_to_instrument_coordinate(mp):
    """Convert FSI ACM measurements from path to instrument coordinates.

    Parameters
    ----------
    mp : xr.Dataset
        MP dataset.

    Returns
    -------
    mp : xr.Dataset
        MP dataset with variables Vx, Vy, Vz1, and Vz2 added.

    Notes
    -----
    Vab=+X, Vcd=+Y, Vef=-X, and Vgh=-Y.
    These correspond to columns 6 to 9 (starting counts at 1) in the raw ACM file.

    Vx is along the major axis of the profiler ellipse (when looking from top),
    Vy perpendicular to this. Vz1 is the upward z, Vz2 the downward z. When the
    profiler is moving upwards (corresponding to negative velocity measurement
    as the water appears to be moving downwards) the good/upstream measurement
    is Vz1 and vice versa.
    """

    mp["Vx"] = (["asnum"], (-(mp.Vab + mp.Vef) / (2 * 0.707)).data)
    mp["Vy"] = (["asnum"], ((mp.Vab - mp.Vef) / (2 * 0.707)).data)
    mp["Vz1"] = (["asnum"], (mp.Vx - mp.Vgh / 0.707).data)
    mp["Vz2"] = (["asnum"], (-mp.Vx + mp.Vcd / 0.707).data)
    return mp


def add_hab(mp, bottom_depth=None):
    if bottom_depth is None:
        bottom_depth = mp.H.median(dim="time").data
    hab = bottom_depth - mp.depth
    mp.coords["hab"] = hab
    mp.hab.attrs["long_name"] = "height above bottom"
    mp.hab.attrs["units"] = "m"
    return mp


def add_overturns(
    mp,
    alpha=0.64,
    dnoise=5e-4,
    roc=0.2,
    N2_method="bulk",
    background_eps=np.nan,
    overturns_from_t=False,
):
    """Add Thorpe scale dissipation to MP dataset.


    Parameters
    ----------
    mp : xr.Dataset
        MP dataset
    alpha : float, optional
        Coefficient relating the Thorpe and Ozmidov scales. Defaults to 0.64.
    dnoise : float
        Density [kg/m^3] or temperature resolution [K].
    roc : float
        Critical value for overturn ratio Roc. Defaults to 0.2.
    N2_method : str
        Method for buoyancy frequency calculation. See mixsea docs for options.
        Defaults to "bulk".
    background_eps : float, optional
        Background value of epsilon applied where no overturns are detected.
        Defaults to NaN.
    overturns_from_t : bool
        Indicates whether to return results based on using density or
        temperature as sorting variable. Defaults to False (density-based
        sorting).

    Returns
    -------
    mp : xr.Dataset
        MP dataset with variables `eps_ot`, `N2_ot`, `Lt_ot`, `Lp_ot`. If using
        temperature as sorting variable, the output variables will have `_t`
        appended, e.g. `eps_ot_t`.

    Notes
    -----
    This is a wrapper for [mixsea](https://mixsea.readthedocs.io)
    """
    eps, N2, Lt, Lp = xr.apply_ufunc(
        _overturn_ufun,
        mp.depth,
        mp.t,
        mp.s,
        mp.lon,
        mp.lat,
        kwargs=dict(
            dnoise=dnoise,
            alpha=alpha,
            roc=roc,
            N2_method=N2_method,
            background_eps=background_eps,
            overturns_from_t=overturns_from_t,
        ),
        input_core_dims=[
            ["depth"],
            ["depth"],
            ["depth"],
            [],
            [],
        ],
        output_core_dims=[
            ["depth"],
            ["depth"],
            ["depth"],
            ["depth"],
        ],
        vectorize=True,  # Loops over 'time' automatically
    )
    eps.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    if overturns_from_t:
        mp["eps_ot_t"] = eps
        mp["N2_ot_t"] = N2
        mp["Lt_ot_t"] = Lt
        mp["Lp_ot_t"] = Lp
    else:
        mp["eps_ot"] = eps
        mp["N2_ot"] = N2
        mp["Lt_ot"] = Lt
        mp["Lp_ot"] = Lp

    return mp


def _overturn_ufun(
    depth,
    t,
    SP,
    lon,
    lat,
    dnoise,
    alpha,
    roc,
    N2_method,
    background_eps,
    overturns_from_t,
):
    """Helper function for add_overturns()

    Parameters
    ----------
    depth :
    t :
    SP :
    lon :
    lat :
    dnoise :
    alpha :
    roc :
    N2_method :
    background_eps :
    overturns_from_t :

    Returns
    -------
    out_eps, out_N2, out_Lt, out_Lp
    """
    # Get shape for output arrays
    out_array = t.copy() * np.nan
    # Select non-nan data
    notnan = np.isfinite(depth) & np.isfinite(t) & np.isfinite(SP)
    depth = depth[notnan]
    t = t[notnan]
    SP = SP[notnan]
    # Do not use the intermediate profile method
    use_ip = False
    # Run overturn calcs using density as sorting variable
    eps, N2, diag = mx.overturn.eps_overturn(
        depth,
        t,
        SP,
        lon,
        lat,
        dnoise=dnoise,
        alpha=alpha,
        Roc=roc,
        N2_method=N2_method,
        background_eps=background_eps,
        use_ip=use_ip,
        return_diagnostics=True,
        overturns_from_t=overturns_from_t,
    )

    out_eps, out_N2, out_Lt, out_Lp = [
        np.full_like(out_array, np.nan, dtype=float) for _ in range(4)
    ]
    # Fill with results
    out_eps[notnan] = eps
    out_N2[notnan] = N2
    # Use flags to get rid of bad overturns (already applied for eps and N2)
    isbad = diag["noise_flag"] | diag["N2_flag"] | diag["Ro_flag"]
    diag["Lt"][isbad] = np.nan
    diag["Lp"][isbad] = np.nan
    out_Lt[notnan] = diag["Lt"]
    out_Lp[notnan] = diag["Lp"]

    return out_eps, out_N2, out_Lt, out_Lp


def add_gsw_variables(mp):
    SA = gsw.SA_from_SP(mp.s, mp.p, mp.attrs["lon"], mp.attrs["lat"])
    mp["SA"] = (("depth", "time"), SA.data)
    mp.SA.attrs["long_name"] = "absolute salinity"
    mp.SA.attrs["units"] = "g/kg"

    CT = gsw.CT_from_t(SA, mp.t, mp.p)
    mp["CT"] = (("depth", "time"), CT.data)
    mp.CT.attrs["long_name"] = "conservative temperature"
    mp.CT.attrs["units"] = "°C"

    P = np.tile(mp.p.transpose(), (mp.time.size, 1)).transpose()
    mp["P"] = (("depth", "time"), P.data)
    mp.P.attrs["long_name"] = "pressure"
    mp.P.attrs["units"] = "dbar"

    return mp


def add_nsquared(mp):
    if "SA" not in mp:
        mp = add_gsw_variables(mp)
    N2, pmid = gsw.Nsquared(mp.SA, mp.CT, mp.P)
    fint = sp.interpolate.interp1d(x=pmid[:, 0], y=N2, axis=0, bounds_error=False)
    N2i = fint(mp.p)
    mp["N2"] = (("depth", "time"), N2i)
    mp.N2.attrs["long_name"] = r"N$^2$"
    mp.N2.attrs["units"] = r"s$^{-2}$"
    return mp


def add_nsquared_smoothed(mp, dp=16):
    n2_all = np.zeros_like(mp.t) * np.nan
    for i in range(mp.time.size):
        try:
            mpp = mp.isel(time=i)
            n2, pout = gv.ocean.nsqfcn(
                mpp.s.data,
                mpp.t.data,
                mpp.P.data,
                p0=0,
                dp=16,
                lon=mp.attrs["lon"],
                lat=mp.attrs["lat"],
            )
            N2 = sp.interpolate.interp1d(pout, n2, bounds_error=False)(mpp.P)
            n2_all[:, i] = N2
        except:
            pass
    mp["N2s"] = (("depth", "time"), n2_all)
    mp.N2s.attrs["long_name"] = r"N$^2$"
    mp.N2s.attrs["units"] = r"s$^{-2}$"
    return mp
