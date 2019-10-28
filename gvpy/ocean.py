#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.ocean with oceanography related functions

'''

import numpy as np
from scipy.signal import filtfilt
from scipy.interpolate import interp1d, NearestNDInterpolator
from scipy import interpolate
import socket
import xarray as xr
import gsw
from gvpy.misc import nearidx2


def nsqfcn(s, t, p, p0, dp, lon, lat):
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
    G = 9.80655
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
    xi = np.where(p >= 0)
    p = p[xi]
    s = s[xi]
    t = t[xi]

    # Exclude nan in t and s
    xi = np.where((~np.isnan(s)) & (~np.isnan(t)))
    p = p[xi]
    s = s[xi]
    t = t[xi]

    # Put out all nan if no good data left
    if ~p.any():
        n2 = np.nan
        pout = np.nan

    # Reverse order of upward profiles
    if p[-1] < p[0]:
        p = p[::-1]
        t = t[::-1]
        s = s[::-1]

    # Low pass filter temp and salinity to match specified dp
    dp_data = np.diff(p)
    dp_med = np.median(dp_data)
    # [b,a]=butter(4,2*dp_med/dp); %causing problems...
    a = 1
    b = np.hanning(2*np.floor(dp/dp_med))
    b = b/np.sum(b)

    tlp = filtfilt(b, a, t)
    slp = filtfilt(b, a, s)
    plp = filtfilt(b, a, p)

    # Check that p is monotonic
    if np.all(np.diff(plp) >= 0):
        pmin = plp[0]
        pmax = plp[-1]

    # # Sort density if opted for
    #   if sort_dens
    #     rho = sw_pden(slp,tlp,plp,plp);
    #     [rhos, si] = sort(rho,'ascend');
    #     tlp = tlp(si);
    #     slp = slp(si);
    #   end

        while p0 <= pmin:
            p0 = p0+dp

        # End points of nsq window
        pwin = np.arange(p0, pmax, dp)
        ft = interp1d(plp, tlp)
        t_ep = ft(pwin)
        fs = interp1d(plp, slp)
        s_ep = fs(pwin)
        # Determine the number of output points
        (npts,) = t_ep.shape

        # Compute pressures at center points
        pout = np.arange(p0+dp/2, np.max(pwin), dp)

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


def tzfcn(CT, z, z0, dz):
    """Calculate vertical temperature gradient for profile of
    conservative temperature and depth.

    The Function: (1) low-pass filters temperature over dp,
                  (2) linearly interpolates t onto depths, z0,
                      z0+dz, z0+2dz, ....,
                  (4) converts differences in temperature into dt/dz
                  (5) returns NaNs if the filtered depth is not
                      monotonic.
                      
    Adapted from nsqfcn from Gregg and Alford.

    Gunnar Voet
    gvoet@ucsd.edu

    Parameters
    ----------
    CT : float
        Conservative temperature (or potential temperature)
    z : float
        Depth
    z0 : float
        Lower bound (start) depth for output values (not important...)
    dz : float
        Depth interval of output data and smoothing length scale

    Returns
    -------
    tz : Vertical temperature gradient dt/dz [deg/m]

    """
    from scipy.signal import filtfilt
    
    # Change notation for t
    t = CT
    
    # keep input z
    zin = z

    # Make sure data has dtype np.ndarray
    if type(z) is not np.ndarray:
        z = np.array(z)
    if type(t) is not np.ndarray:
        t = np.array(t)

    # Exclude nan in t and z
    xi = np.where((~np.isnan(z)) & (~np.isnan(t)))
    z = z[xi]
    t = t[xi]

    # Put out all nan if no good data left
    if ~z.any():
        tz = np.nan
        zout = np.nan

    # Low pass filter temp to match specified dz
    dz_data = np.diff(z)
    dz_med = np.median(dz_data)
    a = 1
    b = np.hanning(2*np.floor(dz/dz_med))
    b = b/np.sum(b)

    tlp = filtfilt(b, a, t)
    zlp = filtfilt(b, a, z)

    # Check that z is monotonic
    if np.all(np.diff(zlp) >= 0):
        zmin = zlp[0]
        zmax = zlp[-1]

        while z0 <= zmin:
            z0 = z0+dz

        # End points of nsq window
        zwin = np.arange(z0, zmax, dz)
        ft = interp1d(zlp, tlp)
        t_ep = ft(zwin)
        # Determine the number of output points
        (npts,) = t_ep.shape

        # Compute depths at center points
        zout = np.arange(z0+dz/2, np.max(zwin), dz)

        # Temperature of upper window pts at output depth
        t_u = t_ep[0:-1]

        # Temperature of lower window pts at output depth
        t_l = t_ep[1:]

        # Compute temperature gradient
        tz = (t_l - t_u)/(dz)
        
        # Interpolate back to original depth
        ftzout = interp1d(zout, tz, bounds_error=False)
        tzout = ftzout(zin)

    else:
        print('  filtered depth not monotonic')
        tz = np.nan
        zout = np.nan

    return tzout


def eps_overturn(P, Z, T, S, lon, lat, dnoise=0.001, pdref=4000, verbose=False):
    '''
    Calculate profile of turbulent dissipation epsilon from structure of a ctd
    profile.
    Currently this takes only one profile and not a matrix e.g. from a whole
    twoyo or ctd section.

    Gunnar Voet
    gvoet@ucsd.edu

    Parameters
    ----------
    P : array-like
        Pressure
    Z : array-like
        Depth
    T : array-like
        In-situ temperature
    S : array-like
        Salinity
    lon : float
        Longitude of observation
    lat : float
        Latitude of observation
    dnoise : float
        Noise level of density in kg/m^3 (default 0.001)
    pdref : float
        Reference pressure for potential density calculation

    Returns
    -------
    out : dict
        Dictionary containing the following results:
      idx : array-like
          Indicator for inversions found (1 for inversion, 0 otherwise)
      Lt : array-like
          Thorpe length scale [m]
      eps : array-like
          Turbulent dissipation [W/kg]
      k : array-like
          Turbulent diffusivity [m^2/s]
      n2 : array-like
          Stratification [s^-2]
      Lo : array-like
          Ozmidov scale
      dtdz : array-like
          Temperature gradient

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
    out['dtdz'] = np.zeros_like(z0)*np.nan
    out['dtdz2'] = np.zeros_like(z0)*np.nan

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

    # Calculate potential density
    sg = gsw.pot_rho_t_exact(SA, t, p, pdref)-1000

    # Create intermediate density profile
    D0 = sg[0]
    sgt = D0-sg[0]
    n = sgt/dnoise
    n = np.fix(n)
    sgi = [D0+n*dnoise]  # first element
    for i in np.arange(1, np.alen(sg), 1):
        sgt = sg[i]-sgi[i-1]
        n = sgt/dnoise
        n = np.fix(n)
        sgi.append(sgi[i-1]+n*dnoise)
    sgi = np.array(sgi)

    # Sort (important to use mergesort here)
    Ds = np.sort(sgi, kind='mergesort')
    Is = np.argsort(sgi, kind='mergesort')

    # Calculate Thorpe length scale
    TH = z[Is]-z
    cumTH = np.cumsum(TH)
    # make sure there are any overturns
    if np.sum(cumTH) > 2:
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

        # Make sure we didn't throw out a zero index in FirstItems
        if len(LastItems) == len(FirstItems)+1:
            if LastItems[0] < FirstItems[0]:
                FirstItems = np.insert(FirstItems, 0, 0)
                if verbose:
                    print('inserting')
                assert len(LastItems) == len(FirstItems)

        # Sort temperature and salinity based on the density sorting index
        # for calculating the buoyancy frequency
        PTs = PT[Is]
        SAs = SA[Is]
        CTs = CT[Is]

        # Loop over detected overturns and calculate Thorpe Scales, N2
        # and dT/dz over the overturn region
        THsc = np.zeros_like(z)*np.nan
        N2 = np.zeros_like(z)*np.nan
        # CN2  = np.ones_like(z)*np.nan
        DTDZ1 = np.zeros_like(z)*np.nan
        DTDZ2 = np.zeros_like(z)*np.nan

        for iostart, ioend in zip(FirstItems, LastItems):
            idx = np.arange(iostart, ioend+1, 1)
            out['idx'][x[idx]] = 1
            sc = np.sqrt(np.mean(np.square(TH[idx])))
            # ctdn2 = np.nanmean(cn2[idx])
            # Buoyancy frequency calculated over the overturn from sorted
            # profiles. Go beyond overturn (I am sure this will cause trouble
            # with the indices at some point).
            n2, Np = gsw.Nsquared(SAs[[iostart-1, ioend+1]],
                                  CTs[[iostart-1, ioend+1]],
                                  p[[iostart-1, ioend+1]], lat)
            # Fill depth range of the overturn with the Thorpe scale
            THsc[idx] = sc
            # Fill depth range of the overturn with N^2
            N2[idx] = n2
            # Fill depth range of the overturn with average 10m N^2
            # CN2[idx]  = ctdn2
            # Fill depth range of the overturn with local temperature gradient
            # Note that numpy's gradient() returns an output vector the same
            # size as the input vector. As we are only providing two input
            # values, we can safely disregard the second output value.
            
            local_dtdz = np.gradient(CTs[[iostart-1, ioend+1]],
                                     z[[iostart-1, ioend+1]])[0]
            DTDZ1[idx] = local_dtdz
                
            if iostart > 0:
                PTov = CTs[iostart-1:ioend+1]
                zov = z[iostart-1:ioend+1]
            else:
                PTov = CTs[iostart:ioend+1]
                zov = z[iostart:ioend+1]

            local_dtdz = (np.min(PTov) - np.max(PTov)) / (np.max(zov) - np.min(zov) )
            DTDZ2[idx] = local_dtdz

        # % Calculate epsilon
        THepsilon = 0.9*THsc**2.0*np.sqrt(N2)**3
        THepsilon[N2 <= 0] = np.nan
        THk = 0.2*THepsilon/N2

        out['eps'][x] = THepsilon
        out['k'][x] = THk
        out['n2'][x] = N2
        out['Lt'][x] = THsc
        out['dtdz'][x] = DTDZ1
        out['dtdz2'][x] = DTDZ2

    return out


def eps_overturn2(P, Z, T, S, lon, lat, dnoise=0.001, pdref=4000):
    '''
    NOTE: This version with sorted temperature for calculation of temperature
    gradient.
    Calculate profile of turbulent dissipation epsilon from structure of a ctd
    profile.
    Currently this takes only one profile and not a matrix e.g. from a whole
    twoyo or ctd section.

    Gunnar Voet
    gvoet@ucsd.edu

    Parameters
    ----------
    P : array-like
        Pressure
    Z : array-like
        Depth
    T : array-like
        In-situ temperature
    S : array-like
        Salinity
    lon : float
        Longitude of observation
    lat : float
        Latitude of observation
    dnoise : float
        Noise level of density in kg/m^3 (default 0.001)
    pdref : float
        Reference pressure for potential density calculation

    Returns
    -------
    out : dict
        Dictionary containing the following results:
      idx : array-like
          Indicator for inversions found (1 for inversion, 0 otherwise)
      Lt : array-like
          Thorpe length scale [m]
      eps : array-like
          Turbulent dissipation [W/kg]
      k : array-like
          Turbulent diffusivity [m^2/s]
      n2 : array-like
          Stratification [s^-2]
      Lo : array-like
          Ozmidov scale
      dtdz : array-like
          Temperature gradient

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
    out['dtdz'] = np.zeros_like(z0)*np.nan

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

    # Calculate potential density
    sg = gsw.pot_rho_t_exact(SA, t, p, pdref)-1000

    # Create intermediate density profile
    D0 = sg[0]
    sgt = D0-sg[0]
    n = sgt/dnoise
    n = np.fix(n)
    sgi = [D0+n*dnoise]  # first element
    for i in np.arange(1, np.alen(sg), 1):
        sgt = sg[i]-sgi[i-1]
        n = sgt/dnoise
        n = np.fix(n)
        sgi.append(sgi[i-1]+n*dnoise)
    sgi = np.array(sgi)

    # Sort (important to use mergesort here)
    Ds = np.sort(sgi, kind='mergesort')
    Is = np.argsort(sgi, kind='mergesort')

    # Sort temperature profile as well for calculation of dT/dz
    # Thetas = np.sort(PT, kind='mergesort')
    # ThIs = np.argsort(PT, kind='mergesort')

    # Calculate Thorpe length scale
    TH = z[Is]-z
    cumTH = np.cumsum(TH)
    # make sure there are any overturns
    if np.sum(cumTH) > 2:

        aa = np.where(cumTH > 2)[0]
        blocks = _consec_blocks(aa, combine_gap=2)

        # Sort temperature and salinity based on the density sorting index
        # for calculating the buoyancy frequency
        PTs = PT[Is]
        SAs = SA[Is]
        CTs = CT[Is]

        # Loop over detected overturns and calculate Thorpe Scales, N2
        # and dT/dz over the overturn region
        THsc = np.zeros_like(z)*np.nan
        N2 = np.zeros_like(z)*np.nan
        # CN2  = np.ones_like(z)*np.nan
        DTDZ = np.zeros_like(z)*np.nan

        for iostart, ioend in zip(blocks[:, 0], blocks[:, 1]):
            idx = np.arange(iostart, ioend, 1)
            out['idx'][x[idx]] = 1
            sc = np.sqrt(np.mean(np.square(TH[idx])))
            # ctdn2 = np.nanmean(cn2[idx])
            # Buoyancy frequency calculated over the overturn from sorted
            # profiles. Go beyond overturn (I am sure this will cause trouble
            # with the indices at some point).
            n2, Np = gsw.Nsquared(SAs[[iostart-1, ioend+1]],
                                  CTs[[iostart-1, ioend+1]],
                                  p[[iostart-1, ioend+1]], lat)
            # Fill depth range of the overturn with the Thorpe scale
            THsc[idx] = sc
            # Fill depth range of the overturn with N^2
            N2[idx] = n2
        
            # Fill depth range of the overturn with local temperature gradient
            # Note that numpy's gradient() returns an output vector the same
            # size as the input vector. As we are only providing two input
            # values, we can safely disregard the second output value.
            # local_dtdz = np.gradient(CTs[[iostart-1, ioend+1]],
                                     # z[[iostart-1, ioend+1]])[0]
            # Calculate temperature gradient based on the minimum/maximum tem-
            # perature range over the overturn, similar to a sorted temperature
            # profile.
            if iostart > 0:
                PTov = CTs[iostart-1:ioend+1]
                zov = z[iostart-1:ioend+1]
            else:
                PTov = CTs[iostart:ioend+1]
                zov = z[iostart:ioend+1]

            local_dtdz = (np.min(PTov) - np.max(PTov)) / (np.max(zov) - np.min(zov) )
            DTDZ[idx] = local_dtdz

        # % Calculate epsilon
        THepsilon = 0.9*THsc**2.0*np.sqrt(N2)**3
        THepsilon[N2 <= 0] = np.nan
        THk = 0.2*THepsilon/N2

        out['eps'][x] = THepsilon
        out['k'][x] = THk
        out['n2'][x] = N2
        out['Lt'][x] = THsc
        out['dtdz'][x] = DTDZ

    return out


def woa_get_ts(llon, llat, plot=0):
    import xarray as xr
    tempfile = '/Users/gunnar/Data/world_ocean_atlas/woa13_decav_t00_04v2.nc'
    saltfile = '/Users/gunnar/Data/world_ocean_atlas/woa13_decav_s00_04v2.nc'
    sigmafile = '/Users/gunnar/Data/world_ocean_atlas/woa13_decav_I00_04.nc'

    dt = xr.open_dataset(tempfile, decode_times=False)
    a = dt.isel(time=0)
    a.reset_coords(drop=True)
    t = a['t_an']
    T = t.sel(lon=llon, lat=llat, method='nearest').values

    ds = xr.open_dataset(saltfile, decode_times=False)
    a = ds.isel(time=0)
    a.reset_coords(drop=True)
    s = a['s_an']
    S = s.sel(lon=llon, lat=llat, method='nearest').values
    depth = s['depth'].data

    dsg = xr.open_dataset(sigmafile, decode_times=False)
    a = dsg.isel(time=0)
    a.reset_coords(drop=True)
    sg = a['I_an']
    sg = sg.sel(lon=llon, lat=llat, method='nearest').values

    if plot:
        # import gvfigure as gvf
        import matplotlib.pyplot as plt
        # fig,ax = gvf.newfig(3,5)
        # plt.plot(T,depth)
        # ax.invert_yaxis()
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.plot(T, depth, 'k')
        ax1.set_xlabel('Temperature [°C]')
        ax1.set_ylabel('Depth [m]')
        ax2.plot(S, depth, 'k')
        ax2.set_xlabel('Salinity')
        ax3.plot(sg, depth, 'k')
        ax3.set_xlabel(r'$\sigma$ [kg/m$^3$]')
        ax1.invert_yaxis()
        f.set_figwidth(7.5)
        f.set_figheight(5)

    return T, S, sg, depth


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
    if len(yday) > 1:
        if len(year) == 1:
            year = np.ones_like(yday)*year
        if len(lon) == 1:
            lon = np.ones_like(yday)*lon
        if len(lat) == 1:
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


def uv2speeddir(u, v):
    """Convert velocity from u,v to speed and direction

    Parameters
    ----------
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
    direction = np.arctan2(v, u)
    return speed, direction


def smith_sandwell(lon='all', lat='all', subsample=0, verbose=0, r15=0):
    """Load Smith & Sandwell bathymetry

    Parameters
    ----------
    lon : float, list or str
        Longitude range. This may either be a single point,
        a list of points, or 'all' (default). If given onne
        point, the nearest ocean depth is returned. For a
        list of locations, the are encompassing all points
        is returned. 'all' returns the whole bathymetry.
    lat : float, list or str
        Latitude. Same options as lon.

    Returns
    -------
    b : xarray DataArray
        Bathymetry in an xarray DataArray using dask for quick access.
    """
    # Load Smith & Sandwell bathymetry as xarray DataArray
    hn = socket.gethostname()
    hn = hn.split(sep='.')[0]
    if r15:
        resolution=15
    else:
        resolution=30
    if verbose:
        print('working on: ' + hn)
    if hn == 'oahu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo{}.grd'.format(resolution)
    elif hn == 'upolu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo{}.grd'.format(resolution)
    elif hn == 'samoa':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_and_sandwell/topo{}.grd'.format(resolution)
    else:
        print('hostname not recognized, assuming we are on oahu for now')
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo{}.grd'.format(resolution)
    if verbose:
        print('Loading bathymetry...')
    b = xr.open_dataarray(nc_file, chunks=1000)
    b['lon'] = np.mod((b.lon+180), 360)-180
    if lon is not 'all':
        # for only one point
        if np.ma.size(lon) == 1 and np.ma.size(lat) == 1:
            lonmask = nearidx2(b.lon.values, lon)
            latmask = nearidx2(b.lat.values, lat)
            b = b.isel(lon=lonmask, lat=latmask)
        # for a range of lon/lat
        else:
            lonmask = ((b.lon > np.nanmin(lon)) & (b.lon < np.nanmax(lon)))
            latmask = ((b.lat > np.nanmin(lat)) & (b.lat < np.nanmax(lat)))
            b = b.isel(lon=lonmask, lat=latmask)
    return b


def smith_sandwell_section(lon, lat, res=1, ext=0):
    """Extract Smith & Sandwell bathymetry along sections
    defined by lon/lat coordinates.

    Parameters
    ----------
    lon : arraylike
        Longitude position along section
    lat : arraylike
        Latitude position along section
    res : float
        Bathymetry resolution
    ext : float
        Extension on both sides in km. Set to 0 for no extension

    Returns
    -------
    out : dict
        Dictionary with output variables
    """
    
    # Extend range of Smith & Sandwell region to account for extension of the
    # line defined by lon/lat. We'll extend by km / 100.
    extra = np.array([-1 * ext, ext]) / 100
    lonr = np.array([np.min(lon), np.max(lon)]) + extra
    latr = np.array([np.min(lat), np.max(lat)]) + extra

    # Load bathymetry
    bathy = smith_sandwell(lonr, latr)

    out = bathy_section(bathy, lon, lat, res, ext)

    return out


def bathy_section(bathy, lon, lat, res=1, ext=0):
    """Extract bathymetry along pathway
    defined by lon/lat coordinates.

    Parameters
    ----------
    bathy : xarray.Dataset or DataArray
        Bathymetry in xarray format, either as DataArray or as Dataset
        with only one data variable. Coordinates may either be x/y or
        lon/lat.
    lon : arraylike
        Longitude position along section
    lat : arraylike
        Latitude position along section
    res : float
        Bathymetry resolution
    ext : float
        Extension on both sides in km. Set to 0 for no extension

    Returns
    -------
    out : dict
        Dictionary with output variables
    """

    # Make sure lon and lat have the same dimensions
    assert lon.shape == lat.shape, 'lat and lon must have the same size'
    # Make sure lon and lat have at least 3 elements
    assert len(lon) > 1 and len(lat) > 1, 'lon/lat must have at least 2 elements'

    # Load bathymetry
    coords = list(bathy.coords.keys())
    if 'x' in coords and 'y' in coords:
        plon = bathy.x
        plat = bathy.y
    elif 'lon' in coords and 'lat' in coords:
        plon = bathy.lon
        plat = bathy.lat
    if isinstance(bathy, xr.DataArray):
        ptopo = bathy.data
    elif isinstance(bathy, xr.Dataset):
        dvar = list(bathy.data_vars.keys())
        assert len(dvar) == 1, 'Bathymetry dataset must have only one data variable'
        ptopo = bathy[dvar[0]].data

    # 2D interpolation function used below. RectBivariateSpline can't deal with
    # NaN's - we'll use NearestNDInterpolator in this case.
    if np.any(np.isnan(ptopo)):
        print('NaN''s present - using NearestNDInterpolator')
        mplon, mplat = np.meshgrid(plon, plat)
        mask = np.isfinite(ptopo)
        intopo = ptopo[mask]
        inlon = mplon[mask]
        inlat = mplat[mask]
        inpoints = np.vstack((inlon, inlat))
        f = NearestNDInterpolator(inpoints.T, intopo)
        NNDI = True
    else:
        f = interpolate.f = interpolate.RectBivariateSpline(plat, plon, ptopo)
        NNDI = False

    # calculate distance between original points
    dist = np.cumsum(gsw.distance(lon, lat, 0)/1000)
    # distance 0 as first element
    dist = np.insert(dist, 0, 0)

    # Extend lon and lat if ext>0
    if ext:
        '''
        Linear fit to start and end points. Do two separate fits if more than
        4 data points are give. Otherwise fit all points together.

        Need to calculate distance first and then scale the lon/lat extension
        with distance.
        '''
        if len(lon) < 5:
            # only one fit for 4 or less data points
            dlo = np.abs(lon[0]-lon[-1])
            dla = np.abs(lat[0]-lat[-1])
            dd = np.abs(dist[0]-dist[-1])
            # fit either against lon or lat, depending on orientation of
            # section
            if dlo > dla:
                bfit = np.polyfit(lon, lat, 1)
                # extension expressed in longitude (scale dist to lon)
                lonext = 1.1*ext/dd*dlo
                if lon[0] < lon[-1]:
                    elon = np.array([lon[0]-lonext, lon[-1]+lonext])
                else:
                    elon = np.array([lon[0]+lonext, lon[-1]-lonext])
                blat = np.polyval(bfit, elon)
                nlon = np.hstack((elon[0], lon, elon[-1]))
                nlat = np.hstack((blat[0], lat, blat[-1]))
            else:
                bfit = np.polyfit(lat, lon, 1)
                # extension expressed in latitude (scale dist to lat)
                latext = 1.1*ext/dd*dla
                if lat[0] < lat[-1]:
                    elat = np.array([lat[0]-latext, lat[-1]+latext])
                else:
                    elat = np.array([lat[0]+latext, lat[-1]-latext])
                blon = np.polyval(bfit, elat)
                nlon = np.hstack((blon[0], lon, blon[-1]))
                nlat = np.hstack((elat[0], lat, elat[-1]))

        else:
            # one fit on each side of the section as it may change direction
            dlo1 = np.abs(lon[0]-lon[2])
            dla1 = np.abs(lat[0]-lat[2])
            dd1 = np.abs(dist[0]-dist[2])
            dlo2 = np.abs(lon[-3]-lon[-1])
            dla2 = np.abs(lat[-3]-lat[-1])
            dd2 = np.abs(dist[-3]-dist[-1])

            # deal with one side first
            if dlo1 > dla1:
                bfit1 = np.polyfit(lon[0:3], lat[0:3], 1)
                lonext1 = 1.1 * ext/dd1*dlo1
                if lon[0] < lon[2]:
                    elon1 = np.array([lon[0]-lonext1, lon[0]])
                else:
                    elon1 = np.array([lon[0]+lonext1, lon[0]])
                elat1 = np.polyval(bfit1, elon1)
            else:
                bfit1 = np.polyfit(lat[0:3], lon[0:3], 1)
                latext1 = 1.1*ext/dd1*dla1
                if lat[0] < lat[2]:
                    elat1 = np.array([lat[0]-latext1, lat[0]])
                else:
                    elat1 = np.array([lat[0]+latext1, lat[0]])
                elon1 = np.polyval(bfit1, elat1)

            # now the other side
            if dlo2 > dla2:
                bfit2 = np.polyfit(lon[-3:], lat[-3:], 1)
                lonext2 = 1.1 * ext/dd2*dlo2
                if lon[-3] < lon[-1]:
                    elon2 = np.array([lon[-1], lon[-1]+lonext2])
                else:
                    elon2 = np.array([lon[-1], lon[-1]-lonext2])
                elat2 = np.polyval(bfit2, elon2)
            else:
                bfit2 = np.polyfit(lat[-3:], lon[-3:], 1)
                latext2 = 1.1*ext/dd2*dla2
                if lat[-3] < lat[-1]:
                    elat2 = np.array([lat[-1], lat[-1]+latext2])
                else:
                    elat2 = np.array([lat[-1], lat[-1]-latext2])
                elon2 = np.polyval(bfit2, elat2)

            # combine everything
            nlon = np.hstack((elon1[0], lon, elon2[1]))
            nlat = np.hstack((elat1[0], lat, elat2[1]))

        lon = nlon
        lat = nlat

    # Original points (but including extension if there are any)
    olat = lat
    olon = lon

    # Interpolated points
    ilat = []
    ilon = []

    # Output dict
    out = {}

    # calculate distance between points
    dist2 = gsw.distance(lon, lat, 0) / 1000
    if np.ndim(dist2)>1:
        dist2 = dist2[0]
    cdist2 = np.cumsum(dist2)
    cdist2 = np.insert(cdist2, 0, 0)

    if res > 0 or ext > 0:
        # Create evenly spaced points between lon and lat
        # for i = 1:length(lon)-1
        for i in np.arange(0, len(lon)-1, 1):

            n = dist2[i] / res

            dlon = lon[i+1]-lon[i]
            if not dlon == 0:
                deltalon = dlon/n
                lons = np.arange(lon[i], lon[i+1], deltalon)
            else:
                lons = np.tile(lon[i], np.int(np.ceil(n)))
            ilon = np.hstack([ilon, lons])

            dlat = lat[i+1]-lat[i]
            if not dlat == 0:
                deltalat = dlat/n
                lats = np.arange(lat[i], lat[i+1], deltalat)
            else:
                lats = np.tile(lat[i], np.int(np.ceil(n)))
            ilat = np.hstack([ilat, lats])

            if i == len(lon)-1:
                ilon = np.append(ilon, olon[-1])
                ilat = np.append(ilat, olat[-1])

            if i == 0:
                odist = np.array([0, dist[i]])
            else:
                odist = np.append(odist, odist[-1]+dist2[i])

        # Evaluate the 2D interpolation function
        if NNDI:
            itopo = f(ilon, ilat)
        else:
            itopo = f.ev(ilat, ilon)
        idist = np.cumsum(gsw.distance(ilon, ilat, 0) / 1000)
        # distance 0 as first element
        idist = np.insert(idist, 0, 0)

        out['ilon'] = ilon
        out['ilat'] = ilat
        out['idist'] = idist
        out['itopo'] = itopo

    if NNDI:
        out['otopo'] = f(olon, olat)
    else:
        out['otopo'] = f.ev(olat, olon)
    out['olat'] = olat
    out['olon'] = olon
    out['odist'] = cdist2

    out['res'] = res
    out['ext'] = ext

    # Extension specific
    if ext:
        # Remove offset in distance between the two bathymetries
        out['olon'] = out['olon'][1:-1]
        out['olat'] = out['olat'][1:-1]
        out['otopo'] = out['otopo'][1:-1]
        # set odist to 0 at initial lon[0]
        offset = out['odist'][1] - out['odist'][0]
        out['odist'] = out['odist'][1:-1] - offset
        out['idist'] = out['idist'] - offset

    return out


def inertial_period(lat):
    Omega = 7.292e-5
    f = 2 * Omega * np.sin(np.deg2rad(lat))
    Ti = 2 * np.pi / f
    Ti = Ti / 3600 / 24
    print('\nInertial period at {:1.2f}° is {:1.2f} days\n'.format(
                                                       float(lat), np.abs(Ti)))
    return Ti


def inertial_frequency(lat):
    Omega = 7.292e-5
    f = 2 * Omega * np.sin(np.deg2rad(lat))
    return f


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


def lonlatstr(lon, lat):
    """
    Generate longitude/latitude strings from position in decimal format,
    for example

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude

    Returns
    -------
    slon : str
        Longitude string
    slat : str
        Latitude str
    """

    if lon > 180:
        lon = lon-360
    if lon < 0:
        EW = 'W'
    else:
        EW = 'E'
    lonmin = np.abs(lon-np.floor(lon))*60
    slon = '{:3d}° {:6.3f}\' {}'.format(int(np.abs(np.floor(lon))),
                                           lonmin, EW)
    slat = 'dummy'

    if lat > 0:
        NS = 'N'
    else:
        NS = 'S'
    latmin = np.abs(lat-np.floor(lat))*60
    slat = '{:3d}° {:6.3f}\' {}'.format(int(np.abs(np.floor(lat))),
                                         latmin, NS)

    return slon, slat


def _consec_blocks(idx=None, combine_gap=0, combine_run=0):
    """
    block_idx = consec_blocks(idx,combine_gap=0, combine_run=0)

    Routine that returns the start and end indexes of the consecutive blocks
    of the index array (idx). The second argument combines consecutive blocks
    together that are separated by <= combine. This is useful when you want
    to perform some action on the n number of data points either side of a
    gap, say, and don't want that action to be effected by a neighbouring
    gap.

    From Glenn Carter, University of Hawaii
    """
    if idx.size == 0:
        return np.array([])

    # Sort the index data and remove any identical points
    idx = np.unique(idx)

    # Find the block boundaries
    didx = np.diff(idx)
    ii = np.concatenate(((didx>1).nonzero()[0], np.atleast_1d(idx.shape[0]-1)))

    # Create the block_idx array
    block_idx = np.zeros((ii.shape[0], 2), dtype=int)
    block_idx[0,:] = [idx[0], idx[ii[0]]]
    for c in range(1, ii.shape[0]):
        block_idx[c,0] = idx[ii[c-1]+1]
        block_idx[c,1] = idx[ii[c]]

    # Find the gap between and combine blocks that are closer together than
    # the combine_gap threshold
    gap = (block_idx[1:,0]-block_idx[0:-1,1])-1
    if np.any(gap <= combine_gap):
        count = 0
        new_block = np.zeros(block_idx.shape, dtype=int)
        new_block[0,0] = block_idx[0,0]
        for ido in range(block_idx.shape[0]-1):
            if gap[ido] > combine_gap:
                new_block[count,1] = block_idx[ido,1]
                count += 1
                new_block[count,0] = block_idx[ido+1,0]
        new_block[count,1] = block_idx[-1,1]
        block_idx = new_block[:count+1,:]

    # Combine any runs that are shorter than the combine_run threshold
    runlength = (block_idx[:,1] - block_idx[:,0])
    if np.any(runlength <= combine_run):
        count = 0
        new_block = np.zeros(block_idx.shape, dtype=int)
        for ido in range(block_idx.shape[0]):
            if runlength[ido] > combine_run:
                new_block[count,:] = block_idx[ido,:]
                count += 1
        block_idx = new_block[:count,:]

    return np.atleast_2d(block_idx)