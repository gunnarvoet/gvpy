#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.signal with functions for signal processing."""

from __future__ import division, print_function

import numpy as np
import scipy as sp
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


def highpassfilter(x, highcut, fs, order=3, axis=-1):
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
    axis : int, optional
        The axis of x to which the filter is applied. Default is -1.

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
    hpx = filtfilt(b, a, x, axis=axis)
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


def psd(
    g,
    dx,
    axis=1,
    ffttype="p",
    detrend=True,
    window="hamming",
    tser_window=None,
    tser_overlap=None,
):
    """
    Compute power spectral density.

    Adapted from Jen MacKinnon.

    Parameters
    ----------
    g : array-like
        Real or complex input data of size [M * 1], [1 * M] or [M * N]
    dx : float
        Distance or time between entries in g
    axis : int, optional
        Axis along which to fft: 0 = columns, 1 = rows (default)
    ffttype : str, optional
        Flag for fft type

        - 'p' for periodic boundary conditions = pure fft (default)

        - 'c' or 's' if the input is a sum of cosines or sines only in which
          case the input is extended evenly or oddly before ffting.

        - 't' for time series, or any other non-exactly periodic series in
          which the data should be windowed and filtered before computing the
          periodogram. In this case you may also specify:

          * tser_window: an integer that gives the length of each window
            the series should be broken up into for filtering and
            computing the periodogram.  Default is length(g).

          * tser_overlap: an integer equal to the lengh of points
            overlap between sucessive windows. Default = tser_window/2,
            which means 50% overlap.
    detrend : bool, optional
        Detrend along dim by removing a linear fit using scipy.detrend().
        Defaults to True.
    window : str, optional
        Window type. Default 'hamming'. See scipy.signal.get_window() for
        window options.

    Returns
    -------
    Pcw : array-like
        Clockwise power spectral density
    Pccw : array-like
        Counter-clockwise power spectral density
    Ptot : array-like
        Total = ccw + cw psd one-sided spectra
    omega: array-like
        Vector of wavenumbers. For plotting use frequency f = omega/(2*pi).
    """
    M0 = g.shape
    g = np.atleast_2d(g)
    if detrend:
        g = sp.signal.detrend(g)

    # FFT and welch act on rows by default. If we want to calculate column-wise,
    # simply transpose the input matrix.
    if axis == 0 & len(M0) > 1:
        g = np.transpose(g)
    M = g.shape

    # If sin or cos transform needed, appropriately extend time series
    if ffttype == "s":
        g_ext = np.concatenate(
            (g, -1 * np.flip(g[:, 1 : int(M[1]) - 1], axis=1)), axis=1
        )
        g = g_ext
        M = g.shape
    if ffttype == "c":
        g_ext = np.concatenate(
            (g, np.flip(g[:, 1 : int(M[1]) - 1], axis=1)), axis=1
        )
        g = g_ext
        M = g.shape

    # Setup frequency vectors
    df = 1 / (M[1] * dx)
    domega = 2 * np.pi * df
    # full frequency output length M
    f_full = np.linspace(0, (M[1] - 1) * df, num=M[1], endpoint=True)
    if np.remainder(M[1], 2) == 0:  # even -> length M/2+1
        f, step = np.linspace(
            0, M[1] * df / 2, num=int(M[1] / 2 + 1), endpoint=True, retstep=True
        )
    else:  # odd -> length (M+1)/2
        f, step = np.linspace(
            0,
            (M[1] - 1) * df / 2,
            num=int((M[1] + 1) / 2),
            endpoint=True,
            retstep=True,
        )
    assert step - df < df / 1000
    omega = 2 * np.pi * f

    # Compute power spectra using fft
    if ffttype in ["p", "c", "s"]:
        P0 = sp.fftpack.fft(g, axis=-1)
        # Normalize by wavenumber in RADIANS
        Pxx = P0 * np.conj(P0) / M[1] / (M[1] * domega)

    if ffttype == "t":
        if tser_window is None:
            # default one segment
            tser_window = np.array(np.fix(M[1]))
        if tser_overlap is None:
            # default 50% overlap
            tser_overlap = np.fix(tser_window / 2)

        f0, Pxx0 = sp.signal.welch(
            g,
            fs=1 / dx,
            axis=-1,
            nperseg=tser_window,
            noverlap=tser_overlap,
            window=window,
            return_onesided=False,
        )
        # One side of 0 is shifted to negative values - need to shift back
        # !!! changing here !!!
        # f0[f0 < 0] = f0[f0 < 0] + 1
        # # Since we've shifted the wavenumber near zero to near one, we need to
        # # extend it fully to one for interpolation to work below. Technically,
        # # this should be the same as interpolating from zero to a very small
        # # value.
        # f0[-1] = 1
        # !!! the above code shifting the frequency vector did not work! below
        # is what amy suggests works.
        f0 = np.fft.fftshift(f0) + np.abs(f0).max()

        Pxx0 = sp.interpolate.interp1d(
            f0, Pxx0, bounds_error=False, axis=-1
        )(f_full)
        Pxx = Pxx0.copy()
        Pxx = Pxx / 2 / np.pi  # normalize by radial wavenumber/frequency
    # Separate into cw and ccw spectra
    if np.remainder(M[1], 2) == 0:
        # even lengths, divide 0 and nyquist freq between ccw and cw
        Pccw = np.concatenate(
            (
                np.reshape(0.5 * Pxx[:, 0], (-1, 1)),
                Pxx[:, 1 : int((M[1]) / 2)],
                np.reshape(0.5 * Pxx[:, int(M[1] / 2) + 1], (-1, 1)),
            ),
            axis=1,
        )
        Pcw = np.concatenate(
            (
                np.reshape(0.5 * Pxx[:, 0], (-1, 1)),
                np.flip(Pxx[:, int(M[1] / 2 + 1) : M[1]], axis=1),
                np.reshape(0.5 * Pxx[:, int(M[1] / 2) + 2], (-1, 1)),
            ),
            axis=1,
        )
        Ptot = Pccw + Pcw
    else:
        # odd lengths, divide 0 freq between ccw and cw
        Pccw = np.concatenate(
            (
                np.reshape(0.5 * Pxx[:, 0], (-1, 1)),
                Pxx[:, 1 : int((M[1] + 1) / 2)],
            ),
            axis=1,
        )
        Pcw = np.concatenate(
            (
                np.reshape(0.5 * Pxx[:, 0], (-1, 1)),
                np.flip(Pxx[:, int(((M[1] + 3) / 2)) - 1 : M[1]], axis=1),
            ),
            axis=1,
        )
        Ptot = Pccw + Pcw

    Ptot = np.squeeze(Ptot)
    Pcw = np.squeeze(Pcw)
    Pccw = np.squeeze(Pccw)

    # transpose back if axis=0
    if axis == 0 & len(M0) > 1:
        Ptot = Ptot.transpose()
        Pcw = Pcw.transpose()
        Pccw = Pccw.transpose()

    return Pcw, Pccw, Ptot, omega


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
