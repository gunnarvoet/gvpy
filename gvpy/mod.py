#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.mod with functions for data collected with instruments built by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at Scripps.

**Note:** This submodule has been superseded by the [modfish](https://modscripps.github.io/modfish/modfish.html) package.
"""

import gsw
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import fft, optimize, signal, stats
import xarray as xr
import pandas as pd

import gvpy as gv


def load_epsi_profile(name):
    """Load one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.

    Notes
    -----
    Note that there are also `load_epsi_raw` and `load_epsi_ctd_raw` for
    reading raw time series.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]

    depth = epsi["z"].squeeze()
    dnum = epsi["dnum"].squeeze()
    eps_co1 = epsi["epsilon_co"].squeeze()[:, 0]
    eps_co2 = epsi["epsilon_co"].squeeze()[:, 1]
    chi1 = epsi["chi"].squeeze()[:, 0]
    chi2 = epsi["chi"].squeeze()[:, 1]

    prof = xr.Dataset(
        data_vars=dict(
            time=(("depth"), gv.time.mattime_to_datetime64(dnum)),
            eps_co1=(("depth"), eps_co1),
            eps_co2=(("depth"), eps_co2),
            chi1=(("depth"), chi1),
            chi2=(("depth"), chi2),
        ),
        coords=dict(depth=(("depth"), depth)),
    )
    for v in [
        "w",
        "t",
        "s",
        "th",
        "pr",
        "sgth",
        "pitch",
        "roll",
        "kvis",
        "epsilon_final",
    ]:
        prof[v] = (("depth"), epsi[v].squeeze())

    prof = prof.rename(dict(roll="rol", epsilon_final="eps_final"))

    start_dnum = dnum[~np.isnan(dnum)]
    start_time = gv.time.mattime_to_datetime64(np.nanmean(start_dnum[:10]))
    prof.attrs["start_time"] = gv.time.datetime64_to_str(start_time, unit="m")
    prof.attrs["lon"] = epsi["longitude"].squeeze()
    prof.attrs["lat"] = epsi["latitude"].squeeze()
    prof.attrs["profile number"] = epsi["profNum"].squeeze()
    # prof = prof.where(~np.isnan(prof.depth), drop=True)
    # new_depth= np.arange(0, 1000.5, 0.5)
    # profi = prof.interp(depth=new_depth)
    return prof


def load_epsi_raw(name):
    """Load raw time series for one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    epsi_raw = epsi["epsi"][0][0]
    dnum = epsi_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=epsi_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), gv.time.mattime_to_datetime64(dnum)),
        ),
    )
    for k in epsi_raw.dtype.names:
        raw[k] = (("time"), epsi_raw[k].squeeze())
    return raw


def load_epsi_raw_mat(file):
    d = scipy.io.loadmat(file)
    epsi = d["epsi"][0, 0]
    dnum = epsi["dnum"].squeeze()
    time = gv.time.mattime_to_datetime64(dnum)
    ds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            s1_volt=(("time"), epsi["s1_volt"].squeeze()),
            s2_volt=(("time"), epsi["s2_volt"].squeeze()),
        ),
    )

    ctd = d["ctd"][0, 0]
    dnum = ctd["dnum"].squeeze()
    time = gv.time.mattime_to_datetime64(dnum)
    cds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            t=(("time"), ctd["T"].squeeze()),
            s=(("time"), ctd["S"].squeeze()),
            c=(("time"), ctd["C"].squeeze()),
            p=(("time"), ctd["P"].squeeze()),
            dzdt=(("time"), ctd["dzdt"].squeeze()),
        ),
    )
    # time = gv.time.mattime_to_datetime64(d.ctd.dnum)
    # ctd = xr.Dataset(
    #     coords=dict(
    #         time=(("time"), time),
    #     ),
    #     data_vars=dict(
    #         p=(("time"), d.ctd.P),
    #         t=(("time"), d.ctd.T),
    #         c=(("time"), d.ctd.C),
    #         s=(("time"), d.ctd.S),
    #     ),
    # )
    ds["p"] = (("time"), cds.p.interp_like(ds).data)
    ds["t"] = (("time"), cds.t.interp_like(ds).data)
    ds["c"] = (("time"), cds.c.interp_like(ds).data)
    ds["s"] = (("time"), cds.s.interp_like(ds).data)
    ds["dzdt"] = (("time"), cds.dzdt.interp_like(ds).data)
    return ds


def load_epsi_ctd_raw(name):
    """Load raw CTD time series for one epsi profile.

    Parameters
    ----------
    name : Path or str
        File path or name.

    Returns
    -------
    ds : xr.Dataset
        Data structure with basic variables.
    """
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    epsi_raw = epsi["epsi"][0][0]
    dnum = epsi_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=epsi_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), gv.time.mattime_to_datetime64(dnum)),
        ),
    )
    for k in epsi_raw.dtype.names:
        raw[k] = (("time"), epsi_raw[k].squeeze())
    return raw
    epsi = scipy.io.loadmat(name)
    epsi = epsi["Profile"][0, 0]
    ctd_raw = epsi["ctd"][0][0]
    dnum = ctd_raw["dnum"].squeeze()
    raw = xr.Dataset(
        data_vars=dict(time_s=ctd_raw["time_s"].squeeze()),
        coords=dict(
            time=(("time"), gv.time.mattime_to_datetime64(dnum)),
        ),
    )
    for k in ctd_raw.dtype.names:
        raw[k] = (("time"), ctd_raw[k].squeeze())
    return raw


def load_epsi_grid(file):
    grd = gv.io.loadmat(file)
    time = gv.time.mattime_to_datetime64(grd.dnum)
    ds = xr.Dataset(
        coords=dict(
            depth=(("depth"), grd.z),
            p=(("depth"), grd.pr),
            time=(("time"), time),
            lon=(("time"), grd.longitude),
            lat=(("time"), grd.latitude),
            profn=(("time"), grd.profNum),
        ),
        data_vars=dict(
            t=(("depth", "time"), grd.t),
            th=(("depth", "time"), grd.th),
            # sgth=(("depth", "time"), grd.sgth - sgth_subtract),
            w=(("depth", "time"), grd.w),
            s=(("depth", "time"), grd.s),
            chi1=(("depth", "time"), grd.chi1),
            chi2=(("depth", "time"), grd.chi2),
            eps1=(("depth", "time"), grd.epsilon_co1),
            eps2=(("depth", "time"), grd.epsilon_co2),
            eps=(("depth", "time"), grd.epsilon_final),
            a1=(("depth", "time"), grd.a1),
            a2=(("depth", "time"), grd.a2),
            a3=(("depth", "time"), grd.a3),
        ),
    )

    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name="absolute salinity", units="kg/m$^3$")
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name="conservative temperature", units="°C")
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r"$\sigma_0$", units="kg/m$^3$")

    ds.p.data = np.float64(ds.p.data)

    ds = add_n2(ds, dp=10)

    dist = gsw.distance(ds.lon, ds.lat)
    cdist = np.cumsum(dist)
    cdist = np.insert(cdist, 0, 0)
    ds.coords["dist"] = (("time"), cdist / 1e3)
    ds.dist.attrs = dict(long_name="distance", units="km")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.th.attrs = dict(long_name=r"$\Theta$", units="°C")
    ds.s.attrs = dict(long_name="salinity", units="psu")
    ds.sgth.attrs = dict(long_name=r"$\sigma_\theta$", units=r"kg/m$^3$")
    ds.eps1.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.eps2.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.eps.attrs = dict(long_name=r"$\epsilon$", units="W/kg")
    ds.chi1.attrs = dict(long_name=r"$\chi$", units=r"K$^2$/s")
    ds.chi2.attrs = dict(long_name=r"$\chi$", units=r"K$^2$/s")
    return ds


def load_fctd_raw_mat(file):
    """Read raw FCTD data at 16 Hz from a single file in the `fctd_mat` processing directory.

    Parameters
    ----------
    file : Path or str
        File path or name of one .mat file in the `fctd_mat` directory.

    Returns
    -------
    ds : xr.Dataset
        Data structure with raw time series data.
    mds : xr.Dataset
        Data structure with raw microconductivity time series data.
    """
    d = gv.io.loadmat(file)

    # fctd
    time = gv.time.mattime_to_datetime64(d.time)
    ds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
            lon=(("time"), d.longitude),
            lat=(("time"), d.latitude),
        ),
        data_vars=dict(
            c=(("time"), d.conductivity),
            t=(("time"), d.temperature),
            p=(("time"), d.pressure),
            bb=(("time"), d.bb),
            chla=(("time"), d.chla),
            fDOM=(("time"), d.fDOM),
            dPdt=(("time"), d.dPdt),
            chi=(("time"), d.chi),
            chi2=(("time"), d.chi2),
            w=(("time"), d.w),
        ),
    )
    ds.p.attrs = dict(long_name="pressure", units="dbar")
    ds.c.attrs = dict(long_name="conductivity", units="mS/cm")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.chi.attrs = dict(long_name=r"$\chi$", units="K$^2$/s")
    ds.chi2.attrs = dict(long_name=r"$\chi_2$", units="K$^2$/s")

    # microconductivity
    microtime = gv.time.mattime_to_datetime64(d.microtime)
    mds = xr.Dataset(
        coords=dict(
            time=(("time"), microtime),
        ),
        data_vars=dict(
            ucon=(("time"), d.ucon),
            ucon_corr=(("time"), d.ucon_corr),
        ),
    )

    return ds, mds


def fctd_mat_combine(files):
    """Read a number of files from fctd_mat directory and combine into one time series.

    Parameters
    ----------
    files : list
        List of files.

    Returns
    -------
    ds : xr.Dataset
        Data structure with raw time series data.
    mds : xr.Dataset
        Data structure with raw microconductivity time series data.
    """
    dsa = [load_fctd_raw_mat(file) for file in files]
    # extract ctd (ds) and microconductivity (mds)
    ds = xr.concat([dsi[0] for dsi in dsa], dim="time")
    mds = xr.concat([dsi[1] for dsi in dsa], dim="time")
    # add pressure to microconductivity
    p_interp = ds.p.interp_like(mds)
    mds["p"] = (("time"), p_interp.data)
    return ds, mds


def load_fctd_raw_time_series(fctd_mat_dir, start, end):
    """Combine data from a number of raw FCTD .mat files in the fctd_mat
    directory.

    Parameters
    ----------
    fctd_mat_dir : pathlib.Path
        `fctd_mat` directory.
    start : np.datetime64 or str
        Start time.
    end : np.datetime64 or str
        End time.

    Returns
    -------
    ds : xr.Dataset
        Raw FCTD time series.
    mds : xr.Dataset
        Raw FCTD microconductivity time series.
    """
    all_files = sorted(fctd_mat_dir.glob("EPSI*.mat"))
    file_times = np.array([parse_filename_datetime(file) for file in all_files])
    if type(start) is str:
        start = np.datetime64(start)
    if type(end) is str:
        end = np.datetime64(end)
    ind = np.flatnonzero((file_times > start) & (file_times < end))
    files = [all_files[i] for i in ind]
    return fctd_mat_combine(files)


def load_fctd_grid(file, what="all"):
    tmp = gv.io.loadmat(file)
    match what:
        case "down":
            grd = tmp["FCTDdown"]
        case "dn":
            grd = tmp["FCTDdown"]
        case "up":
            grd = tmp["FCTDup"]
        case _:
            grd = tmp["FCTDgrid"]
    time = gv.time.mattime_to_datetime64(grd.time)
    ds = xr.Dataset(
        coords=dict(
            depth=(("depth"), grd.depth),
            time=(("time"), time),
            lon=(("time"), np.nanmean(grd.longitude, axis=0)),
            lat=(("time"), np.nanmean(grd.latitude, axis=0)),
            longitude_full=(("depth", "time"), grd.longitude),
            latitude_full=(("depth", "time"), grd.latitude),
        ),
        data_vars=dict(
            t=(("depth", "time"), grd.temperature),
            c=(("depth", "time"), grd.conductivity),
            s=(("depth", "time"), grd.salinity),
            density=(("depth", "time"), grd.density),
            p=(("depth", "time"), grd.pressure),
            # bb=(("depth", "time"), grd.bb),
            # chla=(("depth", "time"), grd.chla),
            chi=(("depth", "time"), grd.chi),
            chi2=(("depth", "time"), grd.chi2),
        ),
    )
    for var in ["drop", "altDist", "w", "bb", "chla"]:
        if var in grd.keys():
            ds[var] = (("depth", "time"), grd[var])

    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name="absolute salinity", units="kg/m$^3$")
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name="conservative temperature", units="°C")
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r"$\sigma_0$", units="kg/m$^3$")

    ds = ds.dropna("depth", how="all")
    mask = ~np.isnat(ds.time)
    ds = ds.sel(time=mask)

    ds = add_n2(ds, dp=10)

    ds.chi.attrs = dict(long_name=r"$\chi_1$", units="K$^2$/s")
    ds.chi2.attrs = dict(long_name=r"$\chi_2$", units="K$^2$/s")
    ds.t.attrs = dict(long_name="temperature", units="°C")
    ds.s.attrs = dict(long_name="salinity", units="psu")
    ds.depth.attrs = dict(long_name="depth", units="m")
    dist = gsw.distance(ds.lon.data, ds.lat.data) / 1e3
    dist = np.insert(np.cumsum(dist), 0, 0)
    ds.coords["dist"] = (("time"), dist)
    ds.dist.attrs = dict(long_name="distance", units="km")
    return ds


def plot_epsi_profile(prof):
    start_str = prof.start_time.replace("T", " ")
    opts = dict(linewidth=1)
    fig, ax = gv.plot.quickfig(c=6, sharey=True, grid=True, fgs=(12, 5))
    ax[0].plot(prof.eps_co1, prof.depth, color="C0", **opts)
    ax[0].plot(prof.eps_co2, prof.depth, color="C6", **opts)
    ax[0].set(
        xscale="log",
        xlim=[1e-11, 1e-6],
        ylabel="depth [m]",
        xlabel="$\\mathrm{log}_{10}(\\epsilon)$ [W/kg]",
        title=f"profile {prof.attrs['profile number']:03d}",
    )
    ax[1].plot(prof.chi1, prof.depth, color="C0", **opts)
    ax[1].plot(prof.chi2, prof.depth, color="C6", **opts)
    ax[1].set(
        xscale="log",
        xlabel="$\\mathrm{log}_{10}(\\chi)$ [K$^2$/s]",
        title=start_str,
    )
    ax[0].invert_yaxis()
    ax[2].plot(prof.w, prof.depth, **opts)
    ax[2].set(xlabel="fall rate [m/s]")
    ax[3].plot(prof.pitch, prof.depth, color="C0", **opts)
    ax[3].plot(prof.rol, prof.depth, color="C6", **opts)
    gv.plot.xsym(ax[3])
    ax[3].set(xlabel="pitch/roll [deg]")
    ax[4].plot(prof.t, prof.depth, **opts)
    ax[4].set(xlabel="temperature [°C]")
    ax[5].plot(prof.s, prof.depth, **opts)
    ax[5].set(xlabel="salinity")
    return ax


def add_n2(ds, dp=10):
    # calculate buoyancy frequency
    ds["n2"] = ds.t.copy() * np.nan
    ds.n2.attrs = dict(
        long_name=r"N$^2$", units=r"s$^{-2}$", info=f"smoothed over {dp} dbar"
    )
    for i in range(len(ds.time)):
        dsi = ds.isel(time=i)
        # dsi = dsi.dropna("depth", how="any", subset=["t", "s", "p"])
        try:
            n2, midp = gv.ocean.nsqfcn(
                dsi.s.data,
                dsi.t.data,
                dsi.p.data,
                p0=0,
                dp=dp,
                lon=dsi.lon.data,
                lat=dsi.lat.data,
            )
            n2i = scipy.interpolate.interp1d(midp, n2, bounds_error=False)(dsi.p.data)
            shape = ds.t.shape
            if len(ds.time) == shape[0]:
                ds.n2.data[i, :] = n2i
            elif len(ds.time) == shape[1]:
                ds.n2.data[:, i] = n2i
        except:
            pass
    return ds


def parse_filename_datetime(file):
    yy, mm, dd, time = file.stem.split("I")[1].split("_")
    dtstr = f"20{yy}-{mm}-{dd} {time[:2]}:{time[2:4]}:{time[4:6]}"
    return np.datetime64(dtstr)


# --------------------------------
# --- T-C correction functions ---
# --------------------------------


def add_tcfit_default(ds):
    """
    Get default values for tc fit range depending on depth of cast.

    Range for tc fit is 200dbar to maximum pressure if the cast is
    shallower than 1000dbar, 500dbar to max pressure otherwise.

    Parameters
    ----------
    ds : xarray.Dataset
            CTD time series data structure

    Returns
    -------
    tcfit : tuple
            Upper and lower limit for tc fit in phase_correct.
    """
    if ds.p.max() > 1000:
        tcfit = [500, ds.p.max().data]
    elif ds.p.max() > 300:
        tcfit = [200, ds.p.max().data]
    else:
        tcfit = [50, ds.p.max().data]
    ds.attrs["tcfit"] = tcfit
    return ds


def atanfit(x, f, Phi, W):
    f = np.arctan(2 * np.pi * f * x[0]) + 2 * np.pi * f * x[1] + Phi
    f = np.matmul(np.matmul(f.transpose(), W**4), f)
    return f


def phase_correct(ds, N=2**6, plot_spectra=False):
    """
    Bring temperature and conductivity in phase.

    Parameters
    ----------
    ds : dtype
            description
    N : int
        Number of points per fit segment

    Returns
    -------
    ds : dtype
            description
    """

    # remove spikes
    # TODO: bring this back in. however, the function fails later on if there
    # are nan's present. Could interpolate over anything that is just a few data points
    # for field in ["t1", "t2", "c1", "c2"]:
    # 	  ib = np.squeeze(np.where(np.absolute(np.diff(data[field].data)) > 0.5))
    # 	  data[field][ib] = np.nan

    # ---Spectral Analysis of Raw Data---
    # 16Hz data from SBE49 (note the difference to 24Hz on the SBE9/11 system)
    dt = 1 / 16
    # number of points per segment
    # N = 2**9 (setting for 24 Hz)
    # N = 2**6 (now providing as function argument)

    # only data within tcfit range.
    ii = np.squeeze(
        np.argwhere(
            (ds.p.data > ds.attrs["tcfit"][0]) & (ds.p.data < ds.attrs["tcfit"][1])
        )
    )
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")
    print(f"{m} segments")
    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # fft of each segment (row). Data are detrended, then windowed.
    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(
        signal.detrend(np.reshape(ds.t.data[i1:i2], newshape=(m, N))) * window
    )
    Ac1 = fft.fft(
        signal.detrend(np.reshape(ds.c.data[i1:i2], newshape=(m, N))) * window
    )

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]

    # Frequency
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    f = f[: int(N / 2)]
    fold = f

    # Spectral Estimates. Note: In Matlab, At1*conj(At1) is not complex anymore.
    # Here, it is still a complex number but the imaginary part is zero.
    # We keep only the real part to stay consistent.
    Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N**2, axis=0))
    Ec1 = 2 * np.real(np.nanmean(Ac1 * np.conj(Ac1) / df / N**2, axis=0))

    # Cross Spectral Estimates
    Ct1c1 = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared Coherence Estimates
    Coht1c1 = np.real(Ct1c1 * np.conj(Ct1c1) / (Et1 * Ec1))

    # Cross-spectral Phase Estimates
    Phit1c1 = np.arctan2(np.imag(Ct1c1), np.real(Ct1c1))

    # ---Determine tau and L---
    # tau is the thermistor time constant (sec)
    # L is the lag of t behind c due to sensor separation (sec)
    # Matrix of weights based on squared coherence.
    W1 = np.diag(Coht1c1)
    # Shift phase by 2*pi for better fit? Or is the change in sign shifting
    # from lag to lead? This is not being done in the ctdproc package, however,
    # the fit looks funky if not folding over by 2*pi.
    Phit1c1[Phit1c1 > 0] = Phit1c1[Phit1c1 > 0] - 2 * np.pi
    # Fit
    x1 = optimize.fmin(func=atanfit, x0=[0, 0], args=(f, Phit1c1, W1), disp=False)

    tau1 = x1[0]
    L1 = x1[1]

    print("tau = {:1.4f}s, lag = {:1.4f}s".format(tau1, L1))

    # ---Apply Phase Correction and LP Filter---
    ii = np.squeeze(np.argwhere(ds.p.data > 1))
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")

    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # Transfer function
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    H1 = (1 + 1j * 2 * np.pi * f * tau1) * np.exp(1j * 2 * np.pi * f * L1)

    # Low Pass Filter
    f0 = 6  # Cutoff frequency (set to 6 for 24Hz; increasing leads to less filtering)
    # (not sure what the exponent does - set to 6 for 24Hz; decreasing to 3
    # leads to lots of noise)
    LP = 1 / (1 + (f / f0) ** 6)

    # Restructure data with overlapping segments.
    # Staggered segments
    vars = [
        "t",
        "c",
        "p",
    ]
    vard = {}
    for v in vars:
        if v in ds:
            vard[v] = np.zeros((2 * m - 1, N))
            vard[v][: 2 * m - 1 : 2, :] = np.reshape(ds[v].data[i1:i2], newshape=(m, N))
            vard[v][1::2, :] = np.reshape(
                ds[v].data[i1 + int(N / 2) : i2 - int(N / 2)],
                newshape=(m - 1, N),
            )

    time = ds.time[i1:i2]
    lon = ds.lon[i1:i2]
    lat = ds.lat[i1:i2]

    # FFTs of staggered segments (each row)
    Ad = {}
    for v in vars:
        if v in ds:
            Ad[v] = fft.fft(vard[v])

    # Corrected Fourier transforms of temperature.
    Ad["t"] = Ad["t"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))

    # LP filter remaining variables
    vars2 = [
        "c",
        "p",
    ]
    for v in vars2:
        if v in ds:
            Ad[v] = Ad[v] * (LP * np.ones((2 * m - 1, 1)))

    # Inverse transforms of corrected temperature
    # and low passed other variables
    Adi = {}
    for v in vars:
        if v in ds:
            Adi[v] = np.real(fft.ifft(Ad[v]))
            Adi[v] = np.squeeze(
                np.reshape(Adi[v][:, int(N / 4) : (3 * int(N / 4))], newshape=(1, -1))
            )

    time = time[int(N / 4) : -int(N / 4)]
    lon = lon[int(N / 4) : -int(N / 4)]
    lat = lat[int(N / 4) : -int(N / 4)]

    # Generate output structure. Copy attributes over.
    out = xr.Dataset(coords={"time": time})
    out.attrs = ds.attrs
    out["lon"] = lon
    out["lat"] = lat
    out["dPdt"] = ds.dPdt
    for v in vars:
        if v in ds:
            out[v] = xr.DataArray(Adi[v], coords=(out.time,))
            out[v].attrs = ds[v].attrs
    out = out.assign_attrs(
        dict(
            tau1=tau1,
            L1=L1,
        )
    )

    # ---Recalculate and replot spectra, coherence and phase---
    t1 = Adi["t"][int(N / 4) : -int(N / 4)]  # Now N elements shorter
    c1 = Adi["c"][int(N / 4) : -int(N / 4)]
    # p = Adi["p"][int(N / 4) : -int(N / 4)]

    m = (i2 - N) / N  # number of segments = dof/2
    m = np.floor(m).astype("int64")
    dof = 2 * m  # Number of degrees of freedom (power of 2)
    df = 1 / (N * dt)  # Frequency resolution at dof degrees of freedom.

    window = signal.windows.triang(N) * np.ones((m, N))
    At1 = fft.fft(signal.detrend(np.reshape(t1, newshape=(m, N))) * window)
    Ac1 = fft.fft(signal.detrend(np.reshape(c1, newshape=(m, N))) * window)

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]
    fn = f[0 : int(N / 2)]

    # Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N**2, axis=0))
    Et1n = 2 * np.nanmean(np.absolute(At1[:, : int(N / 2)]) ** 2, 0) / df / N**2
    Ec1n = 2 * np.nanmean(np.absolute(Ac1[:, : int(N / 2)]) ** 2, 0) / df / N**2

    # Cross Spectral Estimates
    Ct1c1n = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N**2, axis=0)

    # Squared Coherence Estimates
    Coht1c1n = np.real(Ct1c1n * np.conj(Ct1c1n) / (Et1n * Ec1n))
    # 95% confidence bound
    # epsCoht1c1n = np.sqrt(2) * (1 - Coht1c1n) / np.sqrt(Coht1c1n) / np.sqrt(m)
    # epsCoht2c2n = np.sqrt(2) * (1 - Coht2c2n) / np.sqrt(Coht2c2n) / np.sqrt(m)
    # 95% significance level for coherence from Gille notes
    betan = 1 - 0.05 ** (1 / (m - 1))

    # Cross-spectral Phase Estimates
    Phit1c1n = np.arctan2(np.imag(Ct1c1n), np.real(Ct1c1n))
    # 95% error bound
    # epsPhit1c1n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht1c1n) / (dof * Coht1c1n))
    # )
    # epsPhit1c2n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht2c2n) / (dof * Coht2c2n))
    # )
    if plot_spectra:
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(9, 7), constrained_layout=True
        )
        ax0, ax1, ax2, ax3 = ax.flatten()

        ax0.plot(fold, Et1, label="1 uncorrected", color="0.5")
        ax0.plot(fn, Et1n, label="sensor 1")
        ax0.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [$^{\circ}$C$^2$/Hz]",
            title="temperature spectra",
        )
        ax0.plot(
            [fn[10], fn[10]],
            [
                dof * Et1n[10] / stats.distributions.chi2.ppf(0.05 / 2, dof),
                dof * Et1n[10] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
            ],
            "k",
        )
        ax0.legend()

        ax1.plot(fold, Ec1, label="1 uncorrected", color="0.5")
        ax1.plot(fn, Ec1n, label="1")
        ax1.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [mmho$^2$/cm$^2$/Hz]",
            title="conductivity spectra",
        )
        # ax1.plot(
        #     [fn[50], fn[50]],
        #     [
        #         dof * Ec1n[100] / stats.distributions.chi2.ppf(0.05 / 2, dof),
        #         dof * Ec1n[100] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
        #     ],
        #     "k",
        # )

        # Coherence between Temperature and Conductivity
        ax2.plot(fold, Coht1c1, color="0.5")
        ax2.plot(fn, Coht1c1n)

        # ax.plot(fn, Coht1c1 / (1 + 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        # ax.plot(fn, Coht1c1 / (1 - 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        ax2.plot(fn, betan * np.ones(fn.size), "k--")
        ax2.set(
            xlabel="frequency [Hz]",
            ylabel="squared coherence",
            ylim=(-0.1, 1.1),
            title="t/c coherence",
        )

        # Phase between Temperature and Conductivity
        ax3.plot(fold, Phit1c1, color="0.5", marker=".", linestyle="")
        ax3.plot(fn, Phit1c1n, marker=".", linestyle="")
        ax3.set(
            xlabel="frequency [Hz]",
            ylabel="phase [rad]",
            ylim=[-4, 4],
            title="t/c phase",
            # 	  xscale="log",
        )
        ax3.plot(
            fold,
            -np.arctan(2 * np.pi * fold * x1[0]) - 2 * np.pi * fold * x1[1],
            "k--",
        )
        # ax3.plot(
        #     fold,
        #     -np.arctan(2 * np.pi * fold * x2[0]) - 2 * np.pi * fold * x2[1],
        #     "k--",
        # )

    return out


def calc_sal(data):
    # Salinity
    SA, SP = calc_allsal(data.c, data.t, data.p, data.lon, data.lat)

    # Absolute salinity
    data["SA"] = (
        ("time",),
        SA.data,
        {
            "long_name": "absolute salinity",
            "units": "g kg-1",
            "standard_name": "sea_water_absolute_salinity",
        },
    )

    # Practical salinity
    data["s"] = (
        ("time",),
        SP.data,
        {
            "long_name": "practical salinity",
            "units": "",
            "standard_name": "sea_water_practical_salinity",
        },
    )
    return data


def calc_allsal(c, t, p, lon, lat):
    """
    Calculate absolute and practical salinity.

    Wrapper for gsw functions. Converts conductivity
    from S/m to mS/cm if output salinity is less than 5.

    Parameters
    ----------
    c : array-like
        Conductivity. See notes on units above.
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure, dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees
    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity
    SP : array-like
        Practical Salinity
    """
    SP = gsw.SP_from_C(c, t, p)
    if np.nanmean(SP) < 5:
        SP = gsw.SP_from_C(10 * c, t, p)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    return SA, SP


def thermal_mass_correction(ds, alpha=0.03, beta=1 / 7):
    """Correct for delayed conductivity measurement due to thermal mass of
    conductivity cell.


    Notes
    -----
    From SeaBird SeaSoft-Win32 manual:

    Cell Thermal Mass uses a recursive filter to remove conductivity cell
    thermal mass effects from the measured conductivity. In areas with steep
    temperature gradients, the correction is on the order of 0.005 PSU. In
    other areas the correction is negligible. Typical values for alpha and
    1/beta are:
    SBE49   alpha = 0.03   1/beta = 7.0

    Cell Thermal Mass: Formulas
    The algorithm used is:
    a = 2 * alpha / (sample interval * beta + 2)
    b = 1 - (2 * a / alpha)
    dc/dT = 0.1 * (1 + 0.006 * [temperature - 20])
    dT = temperature - previous temperature
    ctm [S/m] = -1.0 * b * previous ctm + a * (dc/dT) * dT
    where
    sample interval is measured in seconds and temperature in °C
    ctm is calculated in S/m
    If the input file contains conductivity in units other than S/m, Cell Thermal
    Mass applies the following scale factors to the calculated ctm:
    ctm [mS/cm] = ctm [S/m] * 10.0
    ctm [μS/cm] = ctm [S/m] * 10000.0
    corrected conductivity = c + ctm

    Matthew's fctd code:
    %Sample frequency - do not change.
    CTpar.freq = 16; % sample rate (Hz) for SBE49

    T=myFCTD.tCorrMHA;
    C=myFCTD.cCorrMHA;

    %This code computes ctm, the conductivity thermal mass error.
    % compute/initialize temp diffs, cond corrections
    dTp = T;
    dTp(2:end) = diff(T);
    dTp(1) = dTp(2);
    dcdt = 0.1 * (1 + 0.006*(T-20)); %This is the expression for dcdt from SBE manual!
    ctm = 0*dTp;
    % a,b
    aa = 2 * CTpar.alfa / (2 + CTpar.beta/CTpar.freq);
    bb = 1 - (2*aa/CTpar.alfa);
    % compute corrections
    for i=2:length(C)
        ctm(i) = -1.0*bb*ctm(i-1) + aa*dcdt(i)*dTp(i);
    end


    MP toolbox code:
    if CTpar.alfa>1e-10 && CTpar.beta>=0
    % compute/initialize temp diffs, cond corrections
    dTp = T;
    dTp(2:end) = diff(T);
    dTp(1) = dTp(2);
    dcdt = 0.1 * (1 + 0.006*(T-20));
    ctm = 0*dTp;  %initialize ctm
    % a,b
    aa = 2 * CTpar.alfa / (2 + CTpar.beta/CTpar.freq);
    bb = 1 - (2*aa/CTpar.alfa);
    % compute corrections
    for ii = 2:length(C)
        ctm(ii) = -1.0*bb*ctm(ii-1) + aa*dcdt(ii)*dTp(ii);
    end
    C = C + ctm;
    end
    """
    ds = ds.copy()

    C = ds.c
    T = ds.t

    # from SBE Data Processing Manual
    # alpha = 0.03
    # beta = 1 / 7.0

    # lueck & picklo
    # alpha = 0.02
    # beta = 0.10

    # beta = 1/tau where tau is relaxation time

    # beta = 1 / 3
    # alpha = 0.02

    if 0:
        # as in SBE manual
        freq = 16
        dTp = T.copy()
        dTp[1:] = np.diff(T)
        dTp[0] = dTp[1]
        dcdt = 0.1 * (1 + 0.006 * (T - 20))
        ctm = np.zeros_like(dTp)
        # % a,b
        aa = 2 * alpha / (2 + beta / freq)
        bb = 1 - (2 * aa / alpha)
        for ii in range(1, len(C)):
            ctm[ii] = -1.0 * bb * ctm[ii - 1] + aa * dcdt[ii] * dTp[ii]
        C = C + ctm
    else:
        # as in lueck & picklo
        freq = 8
        gamma = 0.1
        dTp = T.copy()
        dTp[1:] = np.diff(T)
        dTp[0] = dTp[1]
        ctm = np.zeros_like(dTp)
        # % a,b
        aa = 4 * freq * alpha / beta / (1 + 4 * freq / beta)
        bb = 1 - (2 * aa / alpha)
        for ii in range(1, len(C)):
            ctm[ii] = -1.0 * bb * ctm[ii - 1] + aa * gamma * dTp[ii]
        C = C + ctm
    ds.c.data = C
    return ds


def viscous_heating_temperature_correction(v):
    r"""

    From Ullman & Hebert (2014):
    $$\Delta T = 0.8\times10^{-4}\mathrm{Pr}^{0.5}v^2$$
    where $v$ is the flow speed past the sensor.
    The Prandtl number is the ratio of momentum diffusivity and thermal
    diffusivity $\alpha$:
    $$\mathrm{Pr} = \frac{\nu}{\alpha} = \frac{\nu}{k / (\rho C_p)}$$
    with kinematic viscosity $\nu$, density $\rho$, heat capacity $C_p$, and
    thermal conductivity $k$. $\mathrm{Pr}$ is $\mathcal{O}(10)$ for seawater.
    """
    Pr = 15
    scale = 2
    return scale * 0.8e-4 * Pr**0.5 * v**2


def find_lags(ds):
    c = ds.c.data
    t = ds.t.data
    dpdt = ds.dPdt.data
    time = ds.time.data

    def fit_2d_poly(lags, corrs):
        # Fit the quadratic curve
        coefficients = np.polyfit(lags, corrs, 2)

        # Create a polynomial function
        quadratic_function = np.poly1d(coefficients)

        # Find the vertex of the parabola
        vertex_x = -coefficients[1] / (2 * coefficients[0])
        return vertex_x

    def find_corrs(t, c, tr):
        freq = 16
        ci = np.diff(c[tr] - np.mean(c[tr]))
        ti = np.diff(t[tr] - np.mean(t[tr]))
        correlation = scipy.signal.correlate(
            ci - np.mean(ci), ti - np.mean(ti), mode="full"
        )
        lags = scipy.signal.correlation_lags(len(ci), len(ti), mode="full") * 1 / freq
        lag = np.argmax(np.abs(correlation))
        inds = range(lag - 1, lag + 2)
        return lags[inds], correlation[inds]

    t_lp = gv.signal.lowpassfilter(t, lowcut=1 / 4, fs=1)
    c_lp = gv.signal.lowpassfilter(c, lowcut=1 / 4, fs=1)
    # we'll treat dpdt as vertical velocity for now
    dpdt_lp = gv.signal.lowpassfilter(dpdt, lowcut=1 / 32, fs=1)

    n = len(t)
    print(n, "scans")
    m = 80
    print(n // (m // 2), "segments")

    start_index = np.arange(0, n - m, m // 2)

    start_index = np.arange(0, n - m, m // 2)
    lagi = []
    wi = []
    for si in start_index:
        tr = range(si, si + 80)
        lags, corrs = find_corrs(t_lp, c_lp, tr)
        lag = fit_2d_poly(lags, corrs)
        lagi.append(lag)
        wi.append(np.mean(dpdt[tr]))

    return lagi, wi
