#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Module gvpy.signal with functions for signal processing.

'''

from __future__ import print_function, division
import numpy as np
from scipy.signal import butter, filtfilt


def lowpassfilter(x, lowcut, fs, order=3):
    """
    Lowpass-filter a signal using a butterworth filter.

    Parameters
    ----------
    x : array-like
        time series to be filtered

    lowcut : float
        cut-off frequency

    fs : float
        sampling frequency

    order : int
        order of the filter

    Returns
    -------
    lpx : array-like
        lowpass-filtered time series
    """
    b, a = _butter_lowpass(lowcut, fs, order=order)
    lpx = filtfilt(b, a, x)
    return lpx


def bandpassfilter(x, lowcut, highcut, fs, order=3):
    """
    Bandpass-filter a signal using a butterworth filter.

    Parameters
    ----------
    x : array-like
        time series to be filtered

    lowcut : float
        cut-off low frequency

    highcut : float
        cut-off high frequency

    fs : float
        sampling frequency

    order : int
        order of the filter

    Returns
    -------
    bpx : array-like
        bandpass-filtered time series
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    bpx = filtfilt(b, a, x)
    return bpx


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_lowpass(lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a
