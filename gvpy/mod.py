#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.mod with functions for data collected with instruments built by the
[Multiscale Ocean Dynamics](https://mod.ucsd.edu) group at Scripps."""

import gsw
import numpy as np
import scipy
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
    ds["p"]=(("time"), cds.p.interp_like(ds).data)
    ds["t"]=(("time"), cds.t.interp_like(ds).data)
    ds["c"]=(("time"), cds.c.interp_like(ds).data)
    ds["s"]=(("time"), cds.s.interp_like(ds).data)
    ds["dzdt"]=(("time"), cds.dzdt.interp_like(ds).data)
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
    sgth_subtract = 1000 if np.nanmean(grd.sgth > 900) else 0
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
            sgth=(("depth", "time"), grd.sgth - sgth_subtract),
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
    dist = gsw.distance(ds.lon, ds.lat)
    cdist = np.cumsum(dist)
    cdist = np.insert(cdist, 0, 0)
    ds.coords["dist"] = (("time"), cdist/1e3)
    ds.dist.attrs = dict(long_name='distance', units='km')
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
    """Read raw FCTD data from a single file in the `mat` processing directory.

    Parameters
    ----------
    file : Path or str
        File path or name of one .mat file in the `mat` directory.

    Returns
    -------
    ds : xr.Dataset
        Data structure with unconverted and converted raw data.
    """
    d = gv.io.loadmat(file)
    ctd = d.ctd
    time = gv.time.mattime_to_datetime64(ctd.dnum)
    ds = xr.Dataset(
        coords=dict(
            time=(("time"), time),
        ),
        data_vars=dict(
            c_raw=(("time"), ctd.C_raw),
            t_raw=(("time"), ctd.T_raw),
            p_raw=(("time"), ctd.P_raw),
            s_raw=(("time"), ctd.S_raw),
            c=(("time"), ctd.C),
            t=(("time"), ctd.T),
            s=(("time"), ctd.S),
            p=(("time"), ctd.P),
            dpdt=(("time"), ctd.dPdt),
        ),
    )
    ds.time.attrs = dict(long_name='', units='')
    return ds


def load_fctd_grid(file):
    tmp = gv.io.loadmat(file)
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
            altDist=(("depth", "time"), grd.altDist),
            drop=(("depth", "time"), grd.drop),
            w=(("depth", "time"), grd.w),
        ),
    )
    ds["SA"] = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    ds.SA.attrs = dict(long_name='absolute salinity', units='kg/m$^3$')
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.t, ds.p)
    ds.CT.attrs = dict(long_name='conservative temperature', units='°C')
    ds["sgth"] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sgth.attrs = dict(long_name=r'$\sigma_0$', units='kg/m$^3$')

    ds.chi.attrs = dict(long_name=r'$\chi_1$', units='K$^2$/s')
    ds.chi2.attrs = dict(long_name=r'$\chi_2$', units='K$^2$/s')
    ds.t.attrs = dict(long_name='temperature', units='°C')
    ds.s.attrs = dict(long_name='salinity', units='psu')
    ds.depth.attrs = dict(long_name='depth', units='m')
    dist = gsw.distance(ds.lon.data, ds.lat.data) / 1e3
    dist = np.insert(np.cumsum(dist), 0, 0)
    ds.coords["dist"] = (("time"), dist)
    ds.dist.attrs = dict(long_name='distance', units='km')
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
