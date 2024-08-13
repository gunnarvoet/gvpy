#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module gvpy.trilaterate for oceanographic mooring trilateration."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import netCDF4 # noqa: F401 (need this to avoid bug in pytest)
import xarray as xr
from scipy.optimize import least_squares
import gsw

import gvpy as gv

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


class TrilaterationResult:
    "Holds results from triangulation."

    def __init__(self, lon, lat, error, offset=None):
        """Generate object holding results from trilateration calculation.

        Parameters
        ----------
        lon : float
            Longitude
        lat : float
            Latitude
        error : float
            Error estimate
        offset : float
            Distance to planned location in m
        """

        self.lon = lon
        self.lat = lat
        self.error = error
        self.offset = offset

    def __repr__(self):
        return f"TrilaterationResult(lon={self.lon:7.3f}, lat={self.lat:7.3f}, error={self.error:7.3f}, offset={self.offset:4.1f})"

    def calculate_offset(self, reflon, reflat):
        """Calculate distance to reference location.

        Parameters
        ----------
        reflon : float
            Reference longitude
        reflat : float
            Reference latitude
        """

        self.offset = gsw.distance(
            np.array([self.lon, reflon]),
            np.array([self.lat, reflat]),
            p=0,
        )[0]


class Range:
    "Represents one trilateration sounding."

    def __init__(self, distance, pos, time=None, sn=None):
        """Generate object holding results from one trilateration sounding.

        Parameters
        ----------
        distance : float
            Ranging result in m.
        pos : tuple or xr.Dataset
            Location information. Either a tuple holding lon/lat of the point,
            or a time series of the ship GPS track.
        time : str or np.datetime64, optional
            Time of the sounding. Needs to be provided if pos is ship track.
        sn : int, optional
            Serial number of the acoustic release.
        """

        self.time = np.datetime64(time, "ns")
        self.distance = distance
        self.sn = sn
        # unpack position information or extract from navigational data
        if type(pos) is tuple:
            self.lon, self.lat = pos
        elif type(pos) is xr.Dataset:
            self.find_location(pos)

    def find_location(self, pos: xr.Dataset):
        """Find ship location for sounding time in GPS track.

        Parameters
        ----------
        pos : xr.Dataset
            GPS track in xarray.Dataset with data variables lon and lat and
            coordinate time.
        """
        p = pos.interp(time=self.time)
        self.lon = p.lon.data
        self.lat = p.lat.data

    def __str__(self):
        return f"{self.time}  lon: {self.lon:.5f}  lat: {self.lat:.5f}  distance: {self.distance}m"

    def __repr__(self):
        return f"Range(time={self.time}, lon={self.lon:.5f}, lat={self.lat:.5f}, distance={self.distance})"


class Point:
    "Represents one trilateration point that may consist of 1 to N individual soundings."

    def __init__(
        self,
        mooring,
        distances,
        times=None,
        pos=None,
        nav=None,
    ):
        """Create trilateration point. Provide either positions as lon/lat
        tuples, or times of the soundings and the ship's GPS track for position
        information.

        Parameters
        ----------
        mooring : str
            Mooring name
        distances : list or float
            Soundings (signal travel time converted to distance in meters).
        times : list or str or np.datetime64, optional
            Time or times of soundings.
        pos : tuple
            Longitude and latitude of the sounding.
        nav : GPS track information, optional
            Note: Must be provided if pos is None.
        """
        self.mooring = mooring
        self.distances = np.array(distances)
        self.times = times
        self.pos = pos
        self.nav = nav
        self._input_to_ranges()

        self.lon = np.array([r.lon for r in self.ranges])
        self.lat = np.array([r.lat for r in self.ranges])

    def __str__(self):
        return f"mooring: {self.mooring:>6}\nlon: {self.lon}\nlat: {self.lat}\ndistances: {self.distances}"

    def __repr__(self):
        return f"{self.mooring}\n{self.lon}\n{self.lat}\n{self.distances}"

    def horizontal_distance(self, bottom_depth):
        """Calculate horizontal distance from sounding and bottom depth.

        Parameters
        ----------
        bottom_depth : float
            Bottom depth
        """

        self.hdist = np.sqrt(self.distances**2 - bottom_depth**2)
        self.hdist = np.array([self.hdist]) if self.hdist.size == 1 else self.hdist

    def _input_to_ranges(self):
        """Convert input to Point into Ranges."""
        if self.pos is None:
            if self.distances.size == 1:
                self.ranges = [
                    Range(time=self.times, distance=self.distances, pos=self.nav)
                ]
            else:
                self.ranges = [
                    Range(time=ti, distance=di, pos=self.nav)
                    for ti, di in zip(self.times, self.distances)
                ]
        elif self.times is None:
            if self.distances.size == 1:
                self.ranges = [Range(distance=self.distances, pos=self.pos)]
            else:
                self.ranges = [
                    Range(distance=ti, pos=di)
                    for ti, di in zip(self.distances, self.pos)
                ]


class Trilateration:
    def __init__(
        self,
        mooring,
        plan_lon,
        plan_lat,
        bottom_depth=None,
        topo=None,
        nav=None,
        drop_time=None,
    ):
        self.mooring = mooring
        self.plan_lon = plan_lon
        self.plan_lat = plan_lat
        self.bottom_depth = bottom_depth
        self.points = []
        if topo is None:
            self.add_smith_sandwell()
        else:
            self.topo = topo
        self.add_bathy()
        self.nav = nav
        self.drop_time = drop_time

        if drop_time is not None:
            self.drop_time = np.datetime64(drop_time, "ns")
            self.drop_pos = self.nav.interp(time=self.drop_time)
            deltat = slice(self.drop_time - np.timedelta64(20, "m"), self.drop_time)
            self.drop_approach = self.nav.sel(time=deltat)

    def add_ranges(self, distances, times=None, pos=None):
        if pos is None:
            p = Point(self.mooring, distances=distances, times=times, nav=self.nav)
        elif times is None:
            p = Point(self.mooring, distances=distances, pos=pos)
        p.horizontal_distance(self.bottom_depth)
        self.points.append(p)

    def add_smith_sandwell(self):
        print("no bathymetry provided, using Smith & Sandwell")
        toporange = np.array([-0.02, 0.02])
        latrange = self.plan_lat + toporange
        lonscale = np.cos(np.deg2rad(self.plan_lat))
        lonrange = self.plan_lon + toporange / lonscale
        self.topo = gv.ocean.smith_sandwell(lon=lonrange, lat=latrange)

    def plot_locations(self):
        fig, ax = gv.plot.quickfig(fgs=(3, 3))
        ax.plot(self.plan_lon, self.plan_lat, "rx")
        for p in self.points:
            ax.plot(p.lon, p.lat, "k.")
        return ax

    def trilaterate(self, i=0):
        # select i-th sounding at each point
        lon = [p.lon[i] for p in self.points]
        lat = [p.lat[i] for p in self.points]
        hdist = [p.hdist[i] for p in self.points]
        res = trifun(lon, lat, hdist, planlon=self.plan_lon, planlat=self.plan_lat)
        # add offset
        res.calculate_offset(self.plan_lon, self.plan_lat)
        self.result = res
        # find 2 point solutions
        reserr = []
        ind = [[0, 1], [0, 2], [1, 2]]
        for pair in ind:
            lon = [self.points[x].lon[i] for x in pair]
            lat = [self.points[x].lat[i] for x in pair]
            hdist = [self.points[x].hdist[i] for x in pair]
            res2 = trifun(lon, lat, hdist, planlon=self.plan_lon, planlat=self.plan_lat)
            # calculate distances between 2 point solutions and the full 3 point solution
            res2.calculate_offset(res.lon, res.lat)
            reserr.append(res2)
        self.reserr = reserr
        self.compare_trilateration_solutions()
        self.add_actual_depth()
        return res, reserr

    def compare_trilateration_solutions(self):
        lon = [r.lon for r in self.reserr]
        lat = [r.lat for r in self.reserr]
        lonmin = np.min(lon)
        lonmax = np.max(lon)
        latmin = np.min(lat)
        latmax = np.max(lat)
        if self.result.lon > lonmax or self.result.lon < lonmin:
            print("3-point solution longitude outside of 2-point solutions")
            outside = True
        elif self.result.lat > latmax or self.result.lat < latmin:
            print("3-point solution latitude outside of 2-point solutions")
            outside = True
        else:
            outside = False
        if outside:
            self.result_3point = self.result
            dist = gsw.distance(lon, lat, p=0)
            result = TrilaterationResult(
                lon=np.mean(lon), lat=np.mean(lat), error=np.mean(dist)
            )
            result.calculate_offset(self.plan_lon, self.plan_lat)
            self.result = result

    def add_actual_depth(self):
        depth = self.topo.interp(lon=self.result.lon, lat=self.result.lat).data
        self.result.depth = np.int32(np.round(depth))

    def add_bathy(self, pmlat=0.02):
        lonscale = np.cos(np.deg2rad(self.plan_lat))
        pmlon = pmlat / lonscale
        mask = (
            (self.topo.lon < self.plan_lon + pmlon)
            & (self.topo.lon > self.plan_lon - pmlon)
            & (self.topo.lat < self.plan_lat + pmlat)
            & (self.topo.lat > self.plan_lat - pmlat)
        )
        b = self.topo.where(mask, drop=True)
        self.b = b

    def plot_map(self):
        projection = ccrs.Mercator()
        cmap = "Blues"
        fig, (ax, axr) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(9, 8),
            subplot_kw={"projection": projection},
        )

        mindepth = np.min(self.b)
        maxdepth = np.max(self.b)
        ax.contourf(
            self.b.lon,
            self.b.lat,
            self.b,
            np.arange(mindepth, maxdepth, 20),
            cmap=cmap,
            vmin=mindepth,
            vmax=maxdepth + 20,
            extend="both",
            zorder=2,
            transform=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.25,
            color="0.2",
            alpha=0.2,
            zorder=30,
        )
        gl.ylocator = mticker.MaxNLocator(3)
        gl.xlocator = mticker.MaxNLocator(3)
        gl.top_labels = False
        gl.right_labels = False
        lon_formatter = LONGITUDE_FORMATTER
        lat_formatter = LATITUDE_FORMATTER
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        return ax

    def plot_results(self, pmlat=0.005):
        # b = m.add_bathy(topo, pmlonlat=pmlonlat)
        ax = self.plot_map()
        lonscale = np.cos(np.deg2rad(self.plan_lat))
        pmlon = pmlat / lonscale
        topo_extent = (
            self.plan_lon - pmlon,
            self.plan_lon + pmlon,
            self.plan_lat + pmlat,
            self.plan_lat - pmlat,
        )
        ax.set_extent(topo_extent, crs=ccrs.PlateCarree())
        for p in self.points:
            for lon, lat, radius in zip(p.lon, p.lat, p.hdist):
                gv.maps.plot_watch_circle(lon, lat, radius, ax, ec="0.1", alpha=0.3)
        gv.maps.cartopy_scale_bar(ax, (0.2, 0.1), 100, metres_per_unit=1, unit_name="m")
        ax.plot(
            self.plan_lon,
            self.plan_lat,
            transform=ccrs.PlateCarree(),
            color="r",
            marker="o",
        )
        # res, reserr = self.trilaterate(0)
        ax.plot(
            self.result.lon,
            self.result.lat,
            transform=ccrs.PlateCarree(),
            color="w",
            marker="x",
            zorder=60,
        )
        if hasattr(self, "result_3point"):
            ax.plot(
                self.result_3point.lon,
                self.result_3point.lat,
                transform=ccrs.PlateCarree(),
                color="yellow",
                marker="x",
                zorder=60,
            )
        if self.drop_time is not None:
            ax.plot(
                self.drop_approach.lon,
                self.drop_approach.lat,
                transform=ccrs.PlateCarree(),
                color="C5",
            )
            ax.plot(
                self.drop_pos.lon,
                self.drop_pos.lat,
                transform=ccrs.PlateCarree(),
                marker="d",
                color="orange",
            )
        return ax

    def to_netcdf(self):
        if not hasattr(self, "result"):
            self.trilaterate(i=0)
        m = xr.Dataset(
            coords={"mooring": ("mooring", np.array([self.mooring]))},
            data_vars={
                "lon_planned": ("mooring", np.array([self.plan_lon])),
                "lat_planned": ("mooring", np.array([self.plan_lat])),
                "lon_actual": ("mooring", np.array([self.result.lon])),
                "lat_actual": ("mooring", np.array([self.result.lat])),
                "offset": ("mooring", np.array([self.result.offset])),
                "depth_planned": ("mooring", np.array([self.bottom_depth])),
                "depth_actual": ("mooring", np.array([self.result.depth])),
            },
        )
        return m

    def print_result(self):
        if not hasattr(self, "result"):
            self.trilaterate(i=0)
        lonstr, latstr = gv.ocean.lonlatstr(self.result.lon, self.result.lat)
        s = f"{self.mooring:>6}: {latstr}, {lonstr}, {self.result.depth}m"
        print(s)


def trifun(sndlon, sndlat, sndhdist, planlon=None, planlat=None):
    x1, y1, dist_1 = (sndlon[0], sndlat[0], sndhdist[0])
    x2, y2, dist_2 = (sndlon[1], sndlat[1], sndhdist[1])
    if len(sndlon) > 2:
        x3, y3, dist_3 = (sndlon[2], sndlat[2], sndhdist[2])

    # Define a function that evaluates the equations
    def equations(guess):
        x, y, r = guess
        if len(sndlon) > 2:
            return (
                gsw.distance([x, x1], [y, y1], p=0)[0] - (dist_1 - r),
                gsw.distance([x, x2], [y, y2], p=0)[0] - (dist_2 - r),
                gsw.distance([x, x3], [y, y3], p=0)[0] - (dist_3 - r),
            )
        else:
            return (
                gsw.distance([x, x1], [y, y1], p=0)[0] - (dist_1 - r),
                gsw.distance([x, x2], [y, y2], p=0)[0] - (dist_2 - r),
            )

    # Provide an initial guess - the planned location should be pretty good
    if planlon is not None:
        initial_guess = (planlon, planlat, 0)
    else:
        initial_guess = (sndlon.mean(), sndlat.mean(), 10)

    # Find solution
    res = least_squares(equations, initial_guess)

    # Assign results to data structure
    out = TrilaterationResult(lon=res.x[0], lat=res.x[1], error=res.x[2])

    return out
