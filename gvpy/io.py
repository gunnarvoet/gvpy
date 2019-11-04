#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.misc with in/out functions

'''

from __future__ import print_function, division
import numpy as np
import xarray as xr
from  gvpy.gvimport import mtlb2datetime


def read_sadcp(ncfile):
    sadcp = xr.open_dataset(ncfile)
    
    mdepth = sadcp.depth.median(dim='time')
    sadcp = sadcp.drop('depth')
    # sadcp['depth'] = (['depth_cell'], mdepth)
    sadcp = sadcp.rename_dims({'depth_cell': 'z'})
    sadcp.coords['z'] = (['z'], mdepth)
    # Fix some attributes
    sadcp.z.attrs = dict(long_name='depth', units='m')
    sadcp.u.attrs = dict(long_name='u', units='m/s')
    sadcp.v.attrs = dict(long_name='v', units='m')
    sadcp.time.attrs = dict(long_name='time', units='')
    # Transpose (re-order dimensions)
    sadcp = sadcp.transpose('z', 'time') 
    
    return sadcp


def mat2dataset(m1):
    k = m1.keys()

    varsint = []
    vars1d =  []
    vars2d = []

    for ki in k:
        try:
            tmp = m1[ki].shape
            if len(tmp)==1:
                vars1d.append(ki)
            elif len(tmp)==2:
                vars2d.append(ki)
        except:
            tmp = None
            varsint.append(ki)
    # let's find probable dimensions. usually we have p or z for depth
    if 'z' in k:
        jj = m1['z'].shape[0]
    elif 'p' in k:
        jj = m1['p'].shape[0]
        
    if 'lon' in k:
        if len(m1['lon'].shape) == 1:
            ii = m1['lon'].shape[0]
    elif 'dnum' in k:
        if len(m1['dnum'].shape) == 1:
            ii = m1['dnum'].shape[0]

    out = xr.Dataset(data_vars={'dummy': (['z', 'x'], np.ones((jj, ii)) * np.nan)})
    # get 1d variables
    for v in vars1d:
        if m1[v].shape[0] == ii:
            out[v] = (['x'], m1[v])
        elif m1[v].shape[0] == jj:
            out[v] = (['z'], m1[v])

    # convert the usual suspects into variables
    suspects = ['lon', 'lat', 'p', 'z', 'depth','dep']
    for si in suspects:
        if si in vars1d:
            out.coords[si] = out[si]
            
    # convert time if possible
    for si in ['datenum', 'dtnum', 'dnum']:
        if si in vars1d and np.median(m1[si])>1e5:
            out.coords['time'] = (['x'], mtlb2datetime(m1[si]))
    
    # get 2d variables
    for v in vars2d:
        if m1[v].shape[0]==ii and m1[v].shape[1]==jj:
            out[v] = (['z', 'x'], m1[v])
        elif m1[v].shape[0]==jj and m1[v].shape[1]==ii:
            out[v] = (['z', 'x'], m1[v])
            
    # swap dim x for time if we have a time vector
    if 'time' in out.coords:
        out = out.swap_dims({'x': 'time'})
            
    # drop dummy
    out = out.drop(['dummy'])
    return out