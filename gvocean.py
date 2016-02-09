'''Module gvocean with oceanography related functions

'''

import numpy as np

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
__version__ = "0.1"

def eps_overturn(P,Z,T,S,lon,lat,dnoise=0.001,pdref=4000):
    '''
    Calculate profile of turbulent dissipation epsilon from structure of a ctd
    profile.
    Currently this takes only one profile and not a matrix from a whole twoyo.
    '''
    import numpy as np
    import gsw    
    
    # avoid error due to nan's in conditional statements
    np.seterr(invalid='ignore')
    
    z0   = Z.copy()
    z0   = z0.astype('float')

    # Find non-NaNs
    x = np.where(np.isfinite(T))
    x = x[0]

    # Extract variables without the NaNs
    p   = P[x].copy()
    z   = Z[x].copy()
    z   = z.astype('float')
    t   = T[x].copy()
    s   = S[x].copy()
    # cn2   = ctdn['n2'][x].copy()
    
    SA = gsw.SA_from_SP(s,t,lon,lat)
    CT = gsw.CT_from_t(SA,t,p)
    PT = gsw.pt0_from_t(SA,t,p)

    sg4 = gsw.pot_rho_t_exact(SA,t,p,pdref)-1000

    # Create intermediate density profile
    D0 = sg4[0]
    sgt = D0-sg4[0]
    n = sgt/dnoise
    n = np.round(n)
    sgi = [D0+n*dnoise] # first element
    for i in np.arange(1,np.alen(sg4),1):
        sgt = sg4[i]-sgi[i-1]
        n = sgt/dnoise
        n = np.round(n)
        sgi.append(sgi[i-1]+n*dnoise)
    sgi = np.array(sgi)

    # Sort
    Ds = np.sort(sgi,kind='mergesort')
    Is = np.argsort(sgi,kind='mergesort')
    
    # Calculate Thorpe length scale
    TH = z[Is]-z
    cumTH = np.cumsum(TH)
    
    aa = np.where(cumTH>1)
    aa = aa[0]

    # last index in overturns
    aatmp = aa.copy()
    aatmp = np.append(aatmp,np.nanmax(aa)+10)
    aad = np.diff(aatmp)
    aadi = np.where(aad>1)
    aadi = aadi[0]
    LastItems = aa[aadi].copy()
    
    # first index in overturns
    aatmp = aa.copy()
    aatmp = np.insert(aatmp,0,-1)
    aad = np.diff(aatmp)
    aadi = np.where(aad>1)
    aadi = aadi[0]
    FirstItems = aa[aadi].copy()

    # % Sort temperature and salinity for calculating the buoyancy frequency
    PTs   = PT[Is]
    SAs   = SA[Is]
    CTs   = CT[Is]

    # % Loop through detected overturns
    # % and calculate Thorpe Scales, N2 and dT/dz over the overturn
    # THsc = nan(size(Z));
    THsc = np.zeros_like(z)*np.nan
    N2   = np.zeros_like(z)*np.nan
    # CN2  = np.ones_like(z)*np.nan
    DTDZ = np.zeros_like(z)*np.nan
    
    out = {}
    out['idx'] = np.zeros_like(z0)

    for iostart,ioend in zip(FirstItems,LastItems):
        idx = np.arange(iostart,ioend+1,1)
        out['idx'][x[idx]] = 1
        sc = np.sqrt(np.mean(np.square(TH[idx])))
        # ctdn2 = np.nanmean(cn2[idx])
        # Buoyancy frequency calculated over the overturn from sorted profiles
        # Go beyond overturn (I am sure this will cause trouble with the indices
        # at some point)
        n2,Np = gsw.Nsquared(SAs[[iostart-1,ioend+1]],
                             CTs[[iostart-1,ioend+1]],
                             p[[iostart-1,ioend+1]],lat)
        # Fill depth range of the overturn with the Thorpe scale
        THsc[idx] = sc
        # Fill depth range of the overturn with N^2
        N2[idx]   = n2
        # Fill depth range of the overturn with average 10m N^2
        # CN2[idx]  = ctdn2

    # % Calculate epsilon
    THepsilon        = 0.9*THsc**2.0*np.sqrt(N2)**3
    THepsilon[N2<=0] = np.nan
    THk              = 0.2*THepsilon/N2

    out['eps']    = np.zeros_like(z0)*np.nan
    out['eps'][x] = THepsilon
    out['k']      = np.zeros_like(z0)*np.nan
    out['k'][x]   = THk
    out['n2']     = np.zeros_like(z0)*np.nan
    out['n2'][x]  = N2
    out['Lt']     = np.zeros_like(z0)*np.nan
    out['Lt'][x]  = THsc
    
    return out