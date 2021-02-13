#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.signal with functions for signal processing."""

from __future__ import division, print_function

import numpy as np
from scipy.signal import butter, filtfilt


def lowpassfilter(x, lowcut, fs, order=3):
    """Low-pass filter a signal using a butterworth filter.

    Parameters
    ----------
    x : array-like
        Time series.

    lowcut : float
        Cut-off frequency in units of fs.

    fs : float
        Sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    lpx : array-like
        Low-pass filtered time series.

    Notes
    -----
    For example, if sampling four times per hour, fs=4. A cut-off period of 24
    hours is then expressed as lowcut=1/24. 
    """
    b, a = _butter_lowpass(lowcut, fs, order=order)
    lpx = filtfilt(b, a, x)
    return lpx


def highpassfilter(x, highcut, fs, order=3):
    """High-pass filter a signal using a butterworth filter.

    Parameters
    ----------
    x : array-like
        Time series.

    highcut : float
        Cut-off frequency in units of fs.

    fs : float
        Sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    hpx : array-like
        High-pass filtered time series.

    Notes
    -----
    For example, if sampling four times per hour, fs=4. A cut-off period of 24
    hours is then expressed as highcut=1/24. 
    """
    b, a = _butter_highpass(highcut, fs, order=order)
    hpx = filtfilt(b, a, x)
    return hpx


def bandpassfilter(x, lowcut, highcut, fs, order=3):
    """Band-pass filter a signal using a butterworth filter.

    Parameters
    ----------
    x : array-like
        Time series.

    lowcut : float
        Cut-off low frequency in units of fs.

    highcut : float
        Cut-off high frequency in units of fs.

    fs : float
        Sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    bpx : array-like
        Band-pass filtered time series.

    Notes
    -----
    For example, if sampling four times per hour, fs=4. A cut-off period of 24
    hours is then expressed as highcut=1/24. 
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    bpx = filtfilt(b, a, x)
    return bpx


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _butter_lowpass(lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="lowpass")
    return b, a


def _butter_highpass(highcut, fs, order=3):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a
