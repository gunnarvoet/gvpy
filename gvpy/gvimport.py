#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.gvimport for importing data to python, mostly ®Matlab stuff
for now.
'''

import scipy.io as spio
import numpy as np
import datetime as dt


def gvloadmat(filename):
    '''
    gvloadmat(filename):
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


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
    Convert Matlab datenum format to python datetime.
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
        t1 = day+dayfrac
        if strip_microseconds and strip_seconds:
            t1 = dt.datetime.replace(t1, microsecond=0, second=0)
        elif strip_microseconds:
            t1 = dt.datetime.replace(t1, microsecond=0)
    else:
        day = [dt.datetime.fromordinal(int(tval)) for tval in matlab_datenum]
        dayfrac = [dt.timedelta(days=tval % 1) - dt.timedelta(days=366) for tval in matlab_datenum]
        t1 = [day1+dayfrac1 for day1, dayfrac1 in zip(day, dayfrac)]
        if strip_microseconds and strip_seconds:
            t1 = [dt.datetime.replace(tval, microsecond=0, second=0) for tval in t1]
        elif strip_microseconds:
            t1 = [dt.datetime.replace(t1, microsecond=0) for tval in t1]

    return t1
