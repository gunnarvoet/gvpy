#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.gvimport for importing data to python, mostly ®Matlab stuff
for now.
'''

import scipy.io as spio
import numpy as np
import datetime as dt
from munch import munchify


def gvloadmat(filename, onevar=False):
    '''
    gvloadmat(filename):
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = _check_keys(data)

    if onevar:
        # let's check if there is only one variable in there and return it
        kk = list(out.keys())
        outvars = []
        for k in kk:
            if k[:2] != '__':
                outvars.append(k)
        if len(outvars) == 1:
            print('returning munchfied data structure')
            return munchify(out[outvars[0]])
        else:
            print('found more than one var...')
            return out
    else:
        return out


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
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
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def matlab2datetime(matlab_datenum):
    '''
    matlab2datetime(matlab_datenum):
    Convert Matlab datenum format to python datetime. Only works for single
    timestamps, use mtlb2datetime for vector input.
    '''
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
    return day + dayfrac


def mtlb2datetime(matlab_datenum, strip_microseconds=False,
                  strip_seconds=False):
    '''
    mtlb2datetime(matlab_datenum):
    Convert Matlab datenum format to python datetime.
    This version also works for vector input and strips
    milliseconds if desired.
    '''
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
            t1[i] = np.datetime64('nat')
        t1 = np.array(t1)

    return t1
