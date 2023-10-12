#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Garrett & Munk empirical wave spectrum in its 1981 form based on Munk's
chapter in *Evolution of Physical Oceanography* \[[link to
pdf](https://ocw.mit.edu/ans7870/textbooks/Wunsch/Edited/Chapter9.pdf)\].
Variable names follow Munk's notation.

This code was written by Joern Callies. The original code is at
https://github.com/joernc/GM81. I did not modify the code in any way but include
it here for ease of use as the original code does not come as installable
package. I may also be adding some notes. Many thanks to Joern for sharing his code!

See also [notes that come with Jody Klymak's GM Matlab toolbox](http://jklymak.github.io/GarrettMunkMatlab/).
"""

# Copyright (C) 2016 Joern Callies
#
# This file is part of GM81.
#
# GM81 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GM81 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GM81. If not, see <http://www.gnu.org/licenses/>.

# This module implemets the empirical spectrum of internal waves developed by
# Garrett and Munk, in the incarnation presented in Munk's chapter in Evolution
# of Physical Oceanography, which can be downloaded here:
# http://ocw.mit.edu/resources/res-12-000-evolution-of-physical-oceanography-spring-2007/part-2/wunsch_chapter9.pdf
# The variable names follow Munk's notation.

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gvpy as gv

E = 6.3e-5
"""Energy parameter $E$ (dimensionless).
"""

js = 3
r"""Mode scale number $j^\star$.
"""

jsum = (np.pi*js/np.tanh(np.pi*js)-1)/(2*js**2)
r"""
Sum over a sufficient number of modes such that $j_u \gg j^\star = 3$
$$
\sum_{j=1}^{j_u} (j^2 + j^{\star2})^{-1} = \\frac{\\frac{\pi j^\star}{ \\tanh( \pi j^\star )} -1 }{2 j^{\star2}}
$$
"""

g = 9.81
"""Gravitational acceleration."""

def omg_k_j(k, j, f, N0, b):
    r"""Frequency $\omega$ as a function of horizontal wavenumber $k$ and mode number $j$.

    $$
    \omega = \sqrt{N_0^2 k^2 +f^2(\pi j / b)^2} / (k^2 + (\pi j / b)^2)
    $$
    """
    return np.sqrt((N0**2*k**2+f**2*(np.pi*j/b)**2)/(k**2+(np.pi*j/b)**2))

def k_omg_j(omg, j, f, N0, b):
    r"""Horizontal wavenumber as a function of frequency omg and mode number j.

    $$
    k = \sqrt{\\frac{\omega^2 -f^2}{N_0^2 - \omega^2}} \\frac{\pi j}{b}
    $$
    """
    return np.sqrt((omg**2-f**2)/(N0**2-omg**2))*np.pi*j/b

def B(omg, f):
    r"""Munk's $B(\omega)$ describing the frequency distribution.

    $$
    B(\omega) = 2 \pi^{-1} f \omega^{-1} (\omega^2 - f^2)^{-1/2}
    $$
    """
    return 2/np.pi*f/omg/np.sqrt(omg**2-f**2)

def H(j):
    # Munk's H(j) describing the mode distribution
    return 1./(j**2+js**2)/jsum

def E_omg_j(omg, j, f):
    # Munk's E(omg,j)
    return B(omg, f)*H(j)*E

def E_k_j(k, j, f, N, N0, b):
    # Munk's E(omg,j) transformed into hor. wavenumber space:
    # E(k,j) = E(omg,j) domg/dk. The transformation is achieved using the
    # dispersion relation (9.23a) in Munk (1981).
    omg = omg_k_j(k, j, f, N0, b)
    domgdk = (N0**2-omg**2)/omg*k/(k**2+(np.pi*j/b)**2)
    return E_omg_j(omg, j, f)*domgdk

def P_k_j(k, j, f, N, N0, b):
    # potential energy spectrum (N^2 times displacement spectrum) as a function
    # of hor. wavenumber k and mode number j
    omg = omg_k_j(k, j, f, N0, b)
    return b**2*N0*N*(omg**2-f**2)/omg**2*E_k_j(k, j, f, N, N0, b)

def K_k_j(k, j, f, N, N0, b):
    # kinetic energy spectrum as a function of hor. wavenumber k and mode
    # number j
    omg = omg_k_j(k, j, f, N0, b)
    return b**2*N0*N*(omg**2+f**2)/omg**2*E_k_j(k, j, f, N, N0, b)

def eta_k_j(k, j, f, N, N0, b):
    # SSH spectrum as a function of hor. wavenumber k and mode number j
    omg = omg_k_j(k, j, f, N0, b)
    return (omg**2-f**2)**2/(f**2*(omg**2+f**2))*K_k_j(k, j, f, N, N0, b)/k**2*f**2/g**2

def P_omg_j(omg, j, f, N, N0, b):
    # potential energy spectrum (N^2 times displacement spectrum) as a function
    # of frequency omg and mode number j
    return b**2*N0*N*(omg**2-f**2)/omg**2*E_omg_j(omg, j, f)

def K_omg_j(omg, j, f, N, N0, b):
    # kinetic energy spectrum as a function of frequency omg and mode number j
    return b**2*N0*N*(omg**2+f**2)/omg**2*E_omg_j(omg, j, f)

def eta_omg_j(omg, j, f, N, N0, b):
    # SSH spectrum as a function of frequency omg and mode number j
    k = k_omg_j(omg, j, f, N0, b)
    return (omg**2-f**2)**2/(f**2*(omg**2+f**2))*K_omg_j(omg, j, f, N, N0, b)/k**2*f**2/g**2

def sqrt_trapz(kh, S):
    # integrate S/sqrt(kh^2-k^2) over all kh, approximating S as piecewise
    # linear but then performing the integration exactly
    a = kh[:-1]
    b = kh[1:]
    A = S[:-1]
    B = S[1:]
    k = kh[0]
    return np.sum(((A-B)*(np.sqrt(a**2-k**2)-np.sqrt(b**2-k**2))+(a*B-b*A)*np.log((a+np.sqrt(a**2-k**2))/(b+np.sqrt(b**2-k**2))))/(b-a))

def calc_1d(k, S):
    # calculate 1D wavenumber spectrum from 2D isotropic wavenumber spectrum:
    # S1d = 2/pi int_k^infty S2d/sqrt(kh^2-k^2) dkh
    # (The normalization is such that int_0^infty S1d dk = int_0^infty S2d dkh.)
    S1d = np.empty(k.size)
    for i in range(k.size):
        S1d[i] = 2/np.pi*sqrt_trapz(k[i:], S[i:])
    return S1d

def plot_E_omg(N, lat):
    """Plot energy (PE & KE) spectra in frequency space.

    Parameters
    ----------
    N : float
        Buoyancy frequency at depth.
    lat : float
        Latitude (to determine inertial frequency).

    Returns
    -------
    ax
    """
    f = gv.ocean.inertial_frequency(lat)

    # surface-extrapolated buoyancy frequency
    N0 = 5.2e-3

    # e-folding scale of N(z)
    b = 1.3e3

    # frequency
    omg = np.logspace(np.log10(1.01*f), np.log10(N), 401)

    # horizontal wavenumber
    k = 2*np.pi*np.logspace(-6, -2, 401)

    # mode number
    j = np.arange(1, 100)

    # reshape to allow multiplication into 2D array
    Omg = np.reshape(omg, (omg.size,1))
    K = np.reshape(k, (k.size,1))
    J = np.reshape(j, (1,j.size))

    # frequency spectra (KE and PE)
    K_omg_j = gv.gm81.K_omg_j(Omg, J, f, N, N0, b)
    P_omg_j = gv.gm81.P_omg_j(Omg, J, f, N, N0, b)

    # wavenumber spectra (KE and PE)
    K_k_j = gv.gm81.K_k_j(K, J, f, N, N0, b)
    P_k_j = gv.gm81.P_k_j(K, J, f, N, N0, b)

    # sum over modes (j refers to modes)
    K_omg = np.sum(K_omg_j, axis=1)
    P_omg = np.sum(P_omg_j, axis=1)
    K_k = np.sum(K_k_j, axis=1)
    P_k = np.sum(P_k_j, axis=1)

    # plot frequency spectra
    fig, ax = plt.subplots()
    ax.loglog(omg/(2*np.pi), 2*np.pi*K_omg, label='KE')
    ax.loglog(omg/(2*np.pi), 2*np.pi*P_omg, label='PE')
    ax.legend(frameon=False)
    ax.set_title('frequency spectra')
    ax.set_xlabel('frequency (cps)')
    ax.set_ylabel('power spectral density (m$^2$/s$^2$/cps)')
    return ax, omg, K_omg, P_omg

def calc_E_omg(N, lat):
    """Calculate energy (PE & KE) spectra in frequency space.

    Parameters
    ----------
    N : float
        Buoyancy frequency at depth.
    lat : float
        Latitude (to determine inertial frequency).

    Returns
    -------
    E : xr.Dataset
        Dataset with KE and PE in frequency space.
    """
    f = gv.ocean.inertial_frequency(lat)

    # surface-extrapolated buoyancy frequency
    N0 = 5.2e-3

    # e-folding scale of N(z)
    b = 1.3e3

    # frequency
    omg = np.logspace(np.log10(1.01*f), np.log10(N), 401)

    # horizontal wavenumber
    k = 2*np.pi*np.logspace(-6, -2, 401)

    # mode number
    j = np.arange(1, 100)

    # reshape to allow multiplication into 2D array
    Omg = np.reshape(omg, (omg.size,1))
    K = np.reshape(k, (k.size,1))
    J = np.reshape(j, (1,j.size))

    # frequency spectra (KE and PE)
    K_omg_j = gv.gm81.K_omg_j(Omg, J, f, N, N0, b)
    P_omg_j = gv.gm81.P_omg_j(Omg, J, f, N, N0, b)

    # sum over modes (j refers to modes)
    K_omg = np.sum(K_omg_j, axis=1)
    P_omg = np.sum(P_omg_j, axis=1)

    E = xr.Dataset(
        data_vars=dict(
            KE=(("omega"), K_omg),
            PE=(("omega"), P_omg),
        ),
        coords=dict(
            omega=(("omega"), omg)
        ),
    )

    return E
