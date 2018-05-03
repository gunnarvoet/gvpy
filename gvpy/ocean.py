#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.ocean with oceanography related functions

'''

import numpy as np
from scipy.signal import filtfilt
from scipy.interpolate import interp1d, RectBivariateSpline
import socket
import xarray as xr
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

        # Sort temperature and salinity for calculating the buoyancy frequency
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


def smith_sandwell(lon='all', lat='all', subsample=0, verbose=0):
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
    if verbose:
        print('working on: ' + hn)
    if hn == 'oahu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
    elif hn == 'upolu':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
    elif hn == 'samoa':
        nc_file = '/Users/gunnar/Data/bathymetry/smith_and_sandwell/topo30.grd'
    else:
        print('hostname not recognized, assuming we are on oahu for now')
        nc_file = '/Users/gunnar/Data/bathymetry/smith_sandwell/topo30.grd'
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


    # Make sure lon and lat have the same dimensions
    assert lon.shape==lat.shape, 'lat and lon must have the same size'
    # Make sure lon and lat have at least 3 elements
    assert len(lon)>1 and len(lat)>1, 'lon/lat must have at least 2 elements'

    # Load bathymetry
    bathy = smith_sandwell(lon, lat)
#     b = bathy['bathy2']
#     plon = b['lon']
    plon = bathy.lon
#     plat = b['lat']
    plat = bathy.lat
#     plon, plat = np.meshgrid(bathy.lon, bathy.lat)
#     ptopo = -b['merged']
    ptopo = -bathy.data

    # 2D interpolation function used below
    f = interpolate.f = RectBivariateSpline(plat,plon,ptopo)

    # calculate distance between original points
    dist = np.cumsum(gsw.distance(lon,lat,0)/1000)
    # distance 0 as first element
    dist = np.insert(dist,0,0)

    # Extend lon and lat if ext>0
    if ext:
        '''
        Linear fit to start and end points. Do two separate fits if more than
        4 data points are give. Otherwise fit all points together.

        Need to calculate distance first and then scale the lon/lat extension with distance.
        '''
        if len(lon)<5:
            # only one fit for 4 or less data points
            dlo = np.abs(lon[0]-lon[-1])
            dla = np.abs(lat[0]-lat[-1])
            dd  = np.abs(dist[0]-dist[-1])
            # fit either against lon or lat, depending on orientation of section
            if dlo>dla:
                bfit = np.polyfit(lon,lat,1)
                # extension expressed in longitude (scale dist to lon)
                lonext = 1.1*ext/dd*dlo
                if lon[0]<lon[-1]:
                    elon = np.array([lon[0]-lonext,lon[-1]+lonext])
                else:
                    elon = np.array([lon[0]+lonext,lon[-1]-lonext])
                blat = np.polyval(bfit,elon)
                nlon = np.hstack((elon[0],lon,elon[-1]))
                nlat = np.hstack((blat[0],lat,blat[-1]))
            else:
                bfit = np.polyfit(lat,lon,1)
                # extension expressed in latitude (scale dist to lat)
                latext = 1.1*ext/dd*dla
                if lat[0]<lat[-1]:
                    elat = np.array([lat[0]-lonext,lat[-1]+lonext])
                else:
                    elat = np.array([lat[0]+lonext,lat[-1]-lonext])
                blon = np.polyval(bfit,elat)
                nlon = np.hstack((blon[0],lon,blon[-1]))
                nlat = np.hstack((elat[0],lat,elat[-1]))

        else:
            # one fit on each side of the section as it may change direction
            dlo1 = np.abs(lon[0]-lon[2])
            dla1 = np.abs(lat[0]-lat[2])
            dd1  = np.abs(dist[0]-dist[2])
            dlo2 = np.abs(lon[-3]-lon[-1])
            dla2 = np.abs(lat[-3]-lat[-1])
            dd2  = np.abs(dist[-3]-dist[-1])

            # deal with one side first
            if dlo1>dla1:
                bfit1 = np.polyfit(lon[0:3],lat[0:3],1)
                lonext1 = 1.1*ext/dd1*dlo1
                if lon[0]<lon[2]:
                    elon1 = np.array([lon[0]-lonext1,lon[0]])
                else:
                    elon1 = np.array([lon[0]+lonext1,lon[0]])
                elat1 = np.polyval(bfit1,elon1)
            else:
                bfit1 = np.polyfit(lat[0:3],lon[0:3],1)
                latext1 = 1.1*ext/dd1*dla1
                if lat[0]<lat[2]:
                    elat1 = np.array([lat[0]-latext1,lat[0]])
                else:
                    elat1 = np.array([lat[0]+latext1,lat[0]])
                elon1 = np.polyval(bfit1,elat1)

            # now the other side
            if dlo2>dla2:
                bfit2 = np.polyfit(lon[-3:],lat[-3:],1)
                lonext2 = 1.1*ext/dd2*dlo2
                if lon[-3]<lon[-1]:
                    elon2 = np.array([lon[-1],lon[-1]+lonext2])
                else:
                    elon2 = np.array([lon[-1],lon[-1]-lonext2])
                elat2 = np.polyval(bfit2,elon2)
            else:
                bfit2 = np.polyfit(lat[-3:],lon[-3:],1)
                latext2 = 1.1*ext/dd2*dla2
                if lat[-3]<lat[-1]:
                    elat2 = np.array([lat[-1],lat[-1]+latext2])
                else:
                    elat2 = np.array([lat[-1],lat[-1]-latext2])
                elon2 = np.polyval(bfit2,elat2)

            # combine everything
            nlon = np.hstack((elon1[0],lon,elon2[1]))
            nlat = np.hstack((elat1[0],lat,elat2[1]))

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
    dist2 = gsw.distance(lon,lat,0)/1000
    dist2 = dist2[0]
    cdist2 = np.cumsum(dist2)
    cdist2 = np.insert(cdist2,0,0)

    # Create evenly spaced points between lon and lat
    # for i = 1:length(lon)-1
    for i in np.arange(0,len(lon)-1,1):

        n = dist2[i]/res

        dlon = lon[i+1]-lon[i]
        if not dlon==0:
            deltalon = dlon/n
            lons = np.arange(lon[i],lon[i+1],deltalon)
        else:
            lons = np.tile(lon[i],np.ceil(n))
        ilon = np.hstack([ilon,lons])

        dlat = lat[i+1]-lat[i]
        if not dlat==0:
            deltalat = dlat/n
            lats = np.arange(lat[i],lat[i+1],deltalat)
        else:
            lats = np.tile(lat[i],np.ceil(n))
        ilat = np.hstack([ilat,lats])

        if i==len(lon)-1:
            ilon = np.append(ilon,olon[-1])
            ilat = np.append(ilat,olat[-1])

        if i==0:
            odist = np.array([0,dist[i]])
        else:
            odist = np.append(odist,odist[-1]+dist2[i])

    # Evaluate the 2D interpolation function
    itopo = f.ev(ilat,ilon)
    idist = np.cumsum(gsw.distance(ilon,ilat,0)/1000)
    # distance 0 as first element
    idist = np.insert(idist,0,0)

    out['ilon'] = ilon
    out['ilat'] = ilat
    out['idist'] = idist
    out['itopo'] = itopo

    out['otopo'] = f.ev(olat,olon)
    out['olat'] = olat
    out['olon'] = olon
    out['odist'] = cdist2

    out['res'] = res
    out['ext'] = ext

    # Extension specific
    if ext:
        # Remove offset in distance between the two bathymetries
        out['olon']  = out['olon'][1:-1]
        out['olat']  = out['olat'][1:-1]
        out['otopo']    = out['otopo'][1:-1]
        # set odist to 0 at initial lon[0]
        offset = out['odist'][1]-out['odist'][0]
        out['odist'] = out['odist'][1:-1]-offset
        out['idist'] = out['idist']-offset

    return out


def inertial_period(lat):
    Omega = 7.292e-5
    f = 2*Omega*np.sin(np.deg2rad(lat))
    Ti = 2*np.pi/f
    Ti = Ti/3600/24
    print('\nInertial period at {:1.2f}° is {:1.2f} days\n'.format(lat, np.abs(Ti)))
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
