#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.ocean with oceanography related functions

'''

import numpy as np
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
import socket
import xarray as xr


def nsqfcn(s,t,p,p0,dp,lon,lat):
    """Calculate square of buoyancy frequency [rad/s]^2 for profile of
    temperature, salinity and pressure.

    The Function: (1) low-pass filters t,s,p over dp,
                  (2) linearly interpolates  t and s onto pressures, p0,
                      p0+dp, p0+2dp, ....,
                  (3) computes upper and lower potential densities at
                      p0+dp/2, p0+3dp/2,...,
                  (4) converts differences in potential density into nsq
                  (5) returns NaNs if the filtered pressure is not
                      monotonic.

    If you want to have the buoyancy frequency in [cyc/s] then
    calculate sqrt(n2)./(2*pi). For the period in [s] do sqrt(n2).*2.*pi

    Adapted from Gregg and Alford.

    Gunnar Voet
    gvoet@ucsd.edu

    Parameters
    ----------
    s : float
        Salinity
    t : float
        In-situ temperature
    p : float
        Pressures
    p0 : float
        Lower bound (start) pressure for output values (not important...)
    dp : float
        Pressure interval of output data
    lon : float
        Longitude of observation
    lat : float
        Latitude of observation

    Returns
    -------
    n2 : Buoyancy frequency squared in (rad/s)^2
    pout : Pressure vector for n2

    """
    G  = 9.80655
    dz = dp
        # Make sure data comes in rows
    #     if isrow(s); s = s'; end
    #     if isrow(t); t = t'; end
    #     if isrow(p); p = p'; end

    # Make sure data has dtype np.ndarray
    if type(s) is not np.ndarray:
        s = np.array(s)
    if type(p) is not np.ndarray:
        p = np.array(p)
    if type(t) is not np.ndarray:
        t = np.array(t)

    # Delete negative pressures
    xi = np.where(p>=0)
    p = p[xi]
    s = s[xi]
    t = t[xi]

    # Exclude nan in t and s
    xi = np.where((~np.isnan(s)) & (~np.isnan(t)));
    p = p[xi]
    s = s[xi]
    t = t[xi]

    # Put out all nan if no good data left
    if ~p.any():
        n2 = np.nan
        pout = np.nan

    # Reverse order of upward profiles
    if p[-1]<p[0]:
        p = p[::-1]
        t = t[::-1]
        s = s[::-1]

    # Low pass filter temp and salinity to match specified dp
    dp_data = np.diff(p)
    dp_med  = np.median(dp_data)
    # [b,a]=butter(4,2*dp_med/dp); %causing problems...
    a = 1
    b = np.hanning(2*np.floor(dp/dp_med))
    b = b/np.sum(b)

    tlp = filtfilt(b,a,t)
    slp = filtfilt(b,a,s)
    plp = filtfilt(b,a,p)

    # Check that p is monotonic
    if np.all(np.diff(plp)>=0):
        pmin = plp[0]
        pmax = plp[-1]

    # # Sort density if opted for
    #   if sort_dens
    #     rho = sw_pden(slp,tlp,plp,plp);
    #     [rhos, si] = sort(rho,'ascend');
    #     tlp = tlp(si);
    #     slp = slp(si);
    #   end

        while p0<=pmin:
            p0 = p0+dp

        # End points of nsq window
        pwin = np.arange(p0,pmax,dp)
        ft = interp1d(plp,tlp)
        t_ep = ft(pwin)
        fs = interp1d(plp,slp)
        s_ep = fs(pwin)
        # Determine the number of output points
        (npts,) = t_ep.shape

        # Compute pressures at center points
        pout = np.arange(p0+dp/2,np.max(pwin),dp)

        # Compute potential density of upper window pts at output pressures
        sa_u = gsw.SA_from_SP(s_ep[0:-1], t_ep[0:-1], lon, lat)
        pd_u = gsw.pot_rho_t_exact(sa_u, t_ep[0:-1], pwin[0:-1], pout)

        # Compute potential density of lower window pts at output pressures
        sa_l = gsw.SA_from_SP(s_ep[1:], t_ep[1:], lon, lat)
        pd_l = gsw.pot_rho_t_exact(sa_l, t_ep[1:], pwin[1:], pout)

        # Compute buoyancy frequency squared
        n2 = G*(pd_l - pd_u)/(dp*pd_u)

    else:
        print('  filtered pressure not monotonic')
        n2 = np.nan
        pout = np.nan

    return n2, pout


def eps_overturn(P, Z, T, S, lon, lat, dnoise=0.001, pdref=4000):
    '''
    Calculate profile of turbulent dissipation epsilon from structure of a ctd
    profile.
    Currently this takes only one profile and not a matrix from a whole twoyo.
    '''
    import numpy as np
    import gsw

    # avoid error due to nan's in conditional statements
    np.seterr(invalid='ignore')

    z0 = Z.copy()
    z0 = z0.astype('float')

    # populate output dict
    out = {}
    out['idx'] = np.zeros_like(z0)
    out['Lt'] = np.zeros_like(z0)*np.nan
    out['eps'] = np.zeros_like(z0)*np.nan
    out['k'] = np.zeros_like(z0)*np.nan
    out['n2'] = np.zeros_like(z0)*np.nan
    out['Lo'] = np.zeros_like(z0)*np.nan


    # Find non-NaNs
    x = np.where(np.isfinite(T))
    x = x[0]

    # Extract variables without the NaNs
    p = P[x].copy()
    z = Z[x].copy()
    z = z.astype('float')
    t = T[x].copy()
    s = S[x].copy()
    # cn2   = ctdn['n2'][x].copy()

    SA = gsw.SA_from_SP(s, t, lon, lat)
    CT = gsw.CT_from_t(SA, t, p)
    PT = gsw.pt0_from_t(SA, t, p)

    sg4 = gsw.pot_rho_t_exact(SA, t, p, pdref)-1000

    # Create intermediate density profile
    D0 = sg4[0]
    sgt = D0-sg4[0]
    n = sgt/dnoise
    n = np.fix(n)
    sgi = [D0+n*dnoise]  # first element
    for i in np.arange(1, np.alen(sg4), 1):
        sgt = sg4[i]-sgi[i-1]
        n = sgt/dnoise
        n = np.fix(n)
        sgi.append(sgi[i-1]+n*dnoise)
    sgi = np.array(sgi)

    # Sort
    Ds = np.sort(sgi, kind='mergesort')
    Is = np.argsort(sgi, kind='mergesort')

    # Calculate Thorpe length scale
    TH = z[Is]-z
    cumTH = np.cumsum(TH)
    # make sure there are any overturns
    if np.sum(cumTH)>2:
        aa = np.where(cumTH > 1)
        aa = aa[0]

        # last index in overturns
        aatmp = aa.copy()
        aatmp = np.append(aatmp, np.nanmax(aa)+10)
        aad = np.diff(aatmp)
        aadi = np.where(aad > 1)
        aadi = aadi[0]
        LastItems = aa[aadi].copy()

        # first index in overturns
        aatmp = aa.copy()
        aatmp = np.insert(aatmp, 0, -1)
        aad = np.diff(aatmp)
        aadi = np.where(aad > 1)
        aadi = aadi[0]
        FirstItems = aa[aadi].copy()

        # % Sort temperature and salinity for calculating the buoyancy frequency
        PTs = PT[Is]
        SAs = SA[Is]
        CTs = CT[Is]

        # % Loop through detected overturns
        # % and calculate Thorpe Scales, N2 and dT/dz over the overturn
        # THsc = nan(size(Z));
        THsc = np.zeros_like(z)*np.nan
        N2 = np.zeros_like(z)*np.nan
        # CN2  = np.ones_like(z)*np.nan
        DTDZ = np.zeros_like(z)*np.nan

        for iostart, ioend in zip(FirstItems, LastItems):
            idx = np.arange(iostart, ioend+1, 1)
            out['idx'][x[idx]] = 1
            sc = np.sqrt(np.mean(np.square(TH[idx])))
            # ctdn2 = np.nanmean(cn2[idx])
            # Buoyancy frequency calculated over the overturn from sorted profiles
            # Go beyond overturn (I am sure this will cause trouble with the
            # indices at some point)
            n2, Np = gsw.Nsquared(SAs[[iostart-1, ioend+1]],
                                  CTs[[iostart-1, ioend+1]],
                                  p[[iostart-1, ioend+1]], lat)
            # Fill depth range of the overturn with the Thorpe scale
            THsc[idx] = sc
            # Fill depth range of the overturn with N^2
            N2[idx] = n2
            # Fill depth range of the overturn with average 10m N^2
            # CN2[idx]  = ctdn2

        # % Calculate epsilon
        THepsilon = 0.9*THsc**2.0*np.sqrt(N2)**3
        THepsilon[N2 <= 0] = np.nan
        THk = 0.2*THepsilon/N2

        out['eps'][x] = THepsilon
        out['k'][x] = THk
        out['n2'][x] = N2
        out['Lt'][x] = THsc

    return out


def woa_get_ts(llon, llat, plot=0):
    import xarray as xr
    tempfile = '/Users/gunnar/Data/world_ocean_atlas/woa13_decav_t00_04v2.nc'
    saltfile = '/Users/gunnar/Data/world_ocean_atlas/woa13_decav_s00_04v2.nc'

    dt = xr.open_dataset(tempfile, decode_times=False)
    a = dt.isel(time=0)
    a.reset_coords(drop=True)
    t = a['t_mn']
    T = t.sel(lon=llon, lat=llat, method='nearest').values

    ds = xr.open_dataset(saltfile, decode_times=False)
    a = ds.isel(time=0)
    a.reset_coords(drop=True)
    s = a['s_mn']
    S = s.sel(lon=llon, lat=llat, method='nearest').values
    depth = s['depth'].data

    if plot:
        # import gvfigure as gvf
        import matplotlib.pyplot as plt
        # fig,ax = gvf.newfig(3,5)
        # plt.plot(T,depth)
        # ax.invert_yaxis()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(T, depth, 'k')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Depth [m]')
        ax2.plot(S, depth, 'k')
        ax2.set_xlabel('Salinity')
        ax1.invert_yaxis()
        f.set_figwidth(5)
        f.set_figheight(5)

    return T, S, depth

def tpxo_extract(year, yday, lon, lat):
    """Extract tidal velocity and height predictions from TPXO model.

    Based on python software from UH.

    Gunnar Voet
    gvoet@ucsd.edu

    Parameters
    ----------
    year : list(float) or float
        Year
    yday : list(float) or float
        yearday
    lon : list(float) or float
        Longitude
    lat : list(float) or float
        Latitude

    Returns
    -------
    out['u'] : Tidal velocity in east-west direction
    out['v'] : Tidal velocity in north-south direction
    out['h'] : Tidal elevation

    """
    from pytide import model
    # make sure all variables have the same length
    if len(yday)>1:
        if len(year)==1:
            year = np.ones_like(yday)*year
        if len(lon)==1:
            lon = np.ones_like(yday)*lon
        if len(lat)==1:
            lat = np.ones_like(yday)*lat

    tidemod = model('tpxo7.2')
    velu = []
    velv = []
    h = []
    for yy, yd, lo, la in list(zip(year, yday, lon, lat)):
        vel = tidemod.velocity(yy, yd, lo, la)
        velu.append(vel.u.data)
        velv.append(vel.v.data)
        htmp = tidemod.height(yy, yd, lo, la)
        h.append(htmp.h.data)
    velu = np.concatenate(velu)
    velv = np.concatenate(velv)
    h = np.concatenate(h)
    out = {}
    out['v'] = velv
    out['u'] = velu
    out['h'] = h
    return out


def uv2speeddir(u,v):
    """Convert velocity from u,v to speed and direction

    u : float
        East-West velocity
    v : float
        North-South velocity
    
    Returns
    -------
    speed : Velocity amplitude
    direction : Velocity direction from 0 to 360 starting North.

    """

    speed = np.sqrt(u**2+v**2)
    direction = np.arctan2(v,u)

    
def smith_sandwell():
    # Load Smith & Sandwell bathymetry as xarray Dataset
    hn = socket.gethostname()
    hn = hn.split(sep='.')[0]
    print('working on: '+ hn)
    if hn == 'oahu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
    elif hn == 'upolu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
    elif hn == 'samoa':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_and_sandwell/topo30.grd'
    else:
        print('hostname not recognized, assuming we are on oahu for now')
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
    print('Loading bathymetry...')
    b = xr.open_dataset(nc_file)
    return b


def inertial_period(lat):
    Omega = 7.292e-5;
    f     = 2*Omega*np.sin(np.deg2rad(lat))
    Ti    = 2*np.pi/f
    Ti    = Ti/3600/24
    print('\nInertial period at {:1.2f}° is {:1.2f} days\n'.format(lat,np.abs(Ti)))
    return Ti


def woce_climatology(lon=None, lat=None, z=None, std=False):
    """
    Read WOCE climatology as xarray Dataset. Tries to read local copy of the
    dataset, falls back to remote server access if data not accessible locally.
    
    Parameters
    ----------
    lon : list, numpy array (optional)
        min/max of these values define longitude mask
    lat : list, numpy array (optional)
        min/max of these values define latitude mask
    z : list, numpy array (optional)
        min/max of these values define depth mask
    std: bool
        If True, also load standard deviations (optional)

    Returns
    -------
    w : xarray Dataset
        WOCE climatology
    ws : xarray Dataset
        WOCE climatology standard deviation
    
    TODO
    ----
    Implement lon/lat/z masks

    Notes
    -----
    Remote data access at:
    http://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/woce/catalog.html
    More info:
    http://icdc.cen.uni-hamburg.de/1/daten/ocean/woce-climatology.html
    """
    woce_local = '/Users/gunnar/Data/woce_hydrography/wghc_params.nc'
    woce_remote = 'http://icdc.cen.uni-hamburg.de/thredds/dodsC/'+\
                  'ftpthredds/woce/wghc_params.nc'
    woce_std_local = '/Users/gunnar/Data/woce_hydrography/wghc_stddev.nc'
    woce_std_remote = 'http://icdc.cen.uni-hamburg.de/thredds/dodsC/'+\
                      'ftpthredds/woce/wghc_stddev.nc'
    # read data, try locally first, fall back to remote
    try:
        w = xr.open_dataset(woce_local)
    except:
        print('accessing data remotely')
        w = xr.open_dataset(woce_remote)
    if std:
        try:
            ws = xr.open_dataset(woce_std_local)
        except:
            ws = xr.open_dataset(woce_std_remote)
    # change a few variable names for easier access
    rnm = {'ZAX': 'z', 'LON': 'lon', 'LAT': 'lat', 'BOT_DEP': 'depth',
           'PRES': 'p', 'TEMP': 't', 'TPOTEN': 'th', 'SALINITY': 's',
           'OXYGEN': 'o2', 'SIG0': 'sg0', 'SIG2': 'sg2', 'SIG4': 'sg4',
           'GAMMAN': 'gamma'}
    w.rename(rnm, inplace=True)
    if std:
        return w, ws
    else:
        return w