# History

Entries from 2023.12 onward are summarized from commit messages, grouped by
release tag, and list notable changes only.

## Unreleased

Changes that alter existing results or figures:

-   `xr` `plot_spectrum` plots density per cpd, as its axis label says.
    Values were per rad/s, about 13,751 times too large.
-   `xr` `plot_spectrum` sets y-limits and slope guides from the data.
-   `signal.psd` honors `axis=0`, takes the Nyquist term from the correct
    bin for even lengths, no longer returns NaN bins for `ffttype="t"`,
    and returns real arrays.
-   `ocean.nsqfcn(sort=True)` passes pressure to `SA_from_SP`. It used to
    pass temperature, which could change the sort order and N².
-   `plot.png` and `plot.pdf` keep dots in file names and relative
    subdirectories. Only a `.png` or `.pdf` suffix is replaced.
-   `gm81.calc_E_omg`, `gm81.plot_E_omg` and the `plot_spectrum` markers
    use |f|, so southern latitudes work.
-   `xr` `tplot`, `io.mat2dataset` and `maps.add_dist_coords` no longer
    modify their input.
-   `ds.gv` adds `dist` only for 1-D `lon` and `lat` along `time`.

Other fixes:

-   Replaced removed or deprecated numpy, xarray, matplotlib and IPython
    calls. `np.trapezoid` requires numpy 2 or newer.
-   Fixed crashes and ignored arguments in `io`, `maps`, `misc`, `ocean`,
    `plot`, `time`, `trilaterate` and `xr`.
-   `ipynbname` is imported only when needed.
-   Added a test suite that runs without external data.
-   Upgraded pillow, tornado, urllib3, certifi, requests, idna, fsspec,
    fonttools and pygments in `uv.lock` to clear Dependabot alerts.

## 2026.10 (2026-10-28)

-   Added the `stats` module with `median_ratio_ci`,
    `min_detectable_ratio` and `coherent_fraction`.
-   Fixed `nsqfcn` returning NaN for profiles coarser than `dp/2`.

## 2026.09 (2026-10-04)

-   Added `signal.band_lag`, a time lag from the cross-spectral phase
    slope.
-   `plot.png` takes `bbox_inches`.
-   `ocean.nsqfcn` and `ocean.eps_overturn` pass pressure to `SA_from_SP`.
-   `quickfig(yi=False)` inverts every axes.
-   Dropped `ContourSet.collections` for matplotlib 3.10 and newer.
-   Docs build from `uv.lock`. License declared as GPL-3.0-or-later.

## 2026.07 (2026-07-24)

-   Moved to the uv build backend. `mixsea` resolves from git through
    `tool.uv.sources`.
-   `plot_spectrum` works without a latitude.
-   `sampling_period` returns a float for sub-second data.
-   `add_nsquared_smoothed` uses its `dp` argument.
-   Added a colorbar font size helper. `tplot` passes font size through.

## 2025.11 (2025-11-17)

-   `plot_spectrum` shows -1 and -2 slopes.
-   Reads the ACM compass calibration log.
-   Added `uv.lock`. Paths work on both macOS and Linux.
-   Colormaps are available from `cm`.

## 2025.07 (2025-07-15)

-   Switched most of the tooling to uv.
-   Removed `figsave()`.
-   The `.gv` accessor loads dask arrays.

## 2025.06 (2025-06-27)

-   Added `read_acm()`.
-   Fixed `smith_sandwell` when passing a slice.

## 2025.04 (2025-05-17)

-   Added a `sort` option to `nsqfcn`.
-   `png` and `pdf` write metadata to the figure file.
-   Added functions that return the current notebook name and path.
-   Added a mean for datetime64 values and the `pettroff_class` colormap.
-   Added a tick option to `quickfig` and a row option to `subplotlabel`.

## 2024.11 (2024-12-01)

-   Added `quickmap`, a distance function, `cdist`, and km/nm conversions.
-   Added a `verbose` option to `nsqfcn`.
-   Updated FCTD and epsi reading code.

## 2024.09 (2024-09-10)

-   Added `now_utc()`.
-   Fixed Matlab time conversion.
-   Updated `smith_sandwell` and `llplot`.

## 2024.06 (2024-07-02)

-   Added conversion to Unix time.
-   Removed the seabird dependency.
-   Fixed bugs in `io.mat2dataset` and trilateration testing.

## 2024.05 (2024-05-27)

-   Added `add_dist_coords()`, wrappers for `to_netcdf()`, and an xarray
    `duration()`.
-   Improved `tplot` and `zplot`.
-   Fixed bugs in `png` and `pdf` saving.
-   Fixed matplotlib backend switching in notebooks.

## 2024.04 (2024-04-05)

-   Added depth calculation from the hydrostatic equation.

## 2023.12 (2023-12-11)

-   Moved to `pyproject.toml`.
-   Added the loguru logger, `misc.log()`.
-   Added `llplot` and `zplot`.

## 0.2.0 (2020)

-   Refactored a bunch of modules.
-   Moved adcp functions to their own module as they require pycurrents
    to be imported.
-   Made import of cartopy optional.
-   Renamed `figure` to `plot`,
    `colormaps` to `cm`.
-   Removed `gvimport` as everything is now in
    `io`.
-   Added documentation.
