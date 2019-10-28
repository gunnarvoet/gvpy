#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.misc with in/out functions

'''

from __future__ import print_function, division
import numpy as np
import xarray as xr


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