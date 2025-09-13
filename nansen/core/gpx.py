from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from nansen.utils.smooth import smooth


class GpxTrack:
    """
    A class to represent a GPX data frame.
    """

    @classmethod
    def from_points(cls, name: str, points: list[dict]):
        df = pd.DataFrame(points)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        segment = np.array(df["segment_index"])
        latitude = np.array(df["latitude"])
        longitude = np.array(df["longitude"])
        elevation = np.array(df["elevation"])
        time = np.array(
            [datetime.fromtimestamp(ts) for ts in df["time"].astype("int64") / 10**9]
        )
        return cls(
            name,
            segments=segment,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            time=time,
        )

    def __init__(
        self,
        name: str,
        latitude: np.ndarray,
        longitude: np.ndarray,
        elevation: np.ndarray = None,
        time: np.ndarray = None,
        segments: np.ndarray = None,
    ):
        self.name = name
        self.segments = segments
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.time = time

    @property
    def n_points(self):
        return len(self.time)

    @staticmethod
    def _distance(pos1, pos2):
        return geodesic(pos1, pos2).kilometers

    @property
    def seconds(self):
        if self.time is not None:
            return (
                np.vectorize(lambda dt: dt)(self.time - self.time[0])
                .astype("timedelta64[s]")
                .astype(float)
            )

    @property
    def distance(self):
        ds = [0]

        x1s = self.latitude[:-1]
        x2s = self.latitude[1:]

        y1s = self.longitude[:-1]
        y2s = self.longitude[1:]

        for x1, x2, y1, y2 in zip(x1s, x2s, y1s, y2s):
            dd = self._distance((x1, y1), (x2, y2))
            ds.append(dd)

        return np.cumsum(ds)

    @property
    def velocity(self):
        if self.time is None:
            return
        dt = np.diff(self.seconds)
        dd = np.diff(self.distance)
        vs = 3600 * dd / dt
        return self._resample(vs, reference=self.seconds)

    def _resample(self, quantity, reference, period=None):
        """Resample derived quantities (velocity, compass) to time or distance."""
        # midpoints corresponding to derived quantity (e.g. velocity)
        midpts = reference[:-1] + (np.diff(reference) / 2)
        raw_data = (midpts, quantity)

        # linear interpolation to fall back to initial times
        qty_resampled = np.interp(reference, *raw_data, period=period)

        return qty_resampled

    def _shortname_to_column(self, name):
        """shorname to column name in self.data."""
        shortnames = {
            "t": "time",
            "s": "duration",
            "d": "distance",
            "v": "velocity",
            "z": "elevation",
            "c": "compass",
        }
        try:
            cname = shortnames[name]
        except KeyError:
            raise ValueError(f"Invalid short name: {name}. ")

        if cname == "time":
            column = getattr(self, "time")
        else:
            try:
                column = getattr(self, cname)
            except KeyError:
                raise KeyError(f"{cname} Data unavailable in current track. ")

        return {"name": cname, "column": column}

    def plot(self, mode, *args, **kwargs):
        """Plot columns of self.data (use pandas DataFrame plot arguments).

        Parameters
        ----------
        - mode (str): 2 letters that define short names for x and y axis
        - *args: any additional argument for matplotlib ax.plot()
        - **kwargs: any additional keyword argument for matplotlib ax.plot()

        Output
        ------
        - matplotlib axes

        Short names
        -----------
        't': 'time'
        's': 'duration (s)'
        'd': 'distance (km)'
        'v': 'velocity (km/h)'
        'z': 'elevation (m)'
        """
        try:
            xname, yname = mode
        except ValueError:
            raise ValueError(
                f"Invalid plot mode (should be two letters, e.g. 'tv', not {mode}"
            )

        xinfo = self._shortname_to_column(xname)
        xlabel = xinfo["name"]
        x = xinfo["column"]

        yinfo = self._shortname_to_column(yname)
        ylabel = yinfo["name"]
        y = yinfo["column"]

        print(type(x))
        print(type(y))
        fig, ax = plt.subplots()
        ax.plot(x, y, *args, **kwargs)

        if xlabel == "time":
            fig.autofmt_xdate()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def plot_gpx(self, **kwargs):
        fig, ax = plt.subplots()

        ax.scatter(self.longitude, self.latitude, c=self.velocity, **kwargs)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"GPX Track: {self.name}")

        return ax

    def trim_mask(self, mask, interpolate=True):
        """Trim points where speed exceeds max_speed_kmh and interpolate the removed points."""
        if interpolate:
            # Create an index array for all points
            all_indices = np.arange(len(self.time))
            # Create an index array for the points that are kept
            kept_indices = all_indices[mask]

            # If there are no points to interpolate, return the trimmed GpxTrack
            if len(kept_indices) < 2:
                latitude = self.latitude[mask]
                longitude = self.longitude[mask]
                elevation = self.elevation[mask]
                time = self.time[mask]
                return GpxTrack(
                    name=self.name,
                    latitude=latitude,
                    longitude=longitude,
                    elevation=elevation,
                    time=time,
                )

            # Interpolate latitude
            f_lat = interp1d(kept_indices, self.latitude[mask])
            interpolated_latitude = f_lat(all_indices)

            # Interpolate longitude
            f_lon = interp1d(kept_indices, self.longitude[mask])
            interpolated_longitude = f_lon(all_indices)

            # Interpolate elevation
            f_ele = interp1d(kept_indices, self.elevation[mask])
            interpolated_elevation = f_ele(all_indices)

            # Time array remains the same
            interpolated_time = self.time

            return GpxTrack(
                name=self.name,
                latitude=interpolated_latitude,
                longitude=interpolated_longitude,
                elevation=interpolated_elevation,
                time=interpolated_time,
            )
        else:
            # Original trimming logic without interpolation
            latitude = self.latitude[mask]
            longitude = self.longitude[mask]
            elevation = self.elevation[mask]
            time = self.time[mask]
            return GpxTrack(
                name=self.name,
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
                time=time,
            )

    def trim_speed(self, max_speed_kmh, interpolate=True, tolerance=10):
        """Trim points where speed exceeds max_speed_kmh and interpolate the removed points."""
        if self.velocity is None:
            return

        high_speed_indices = np.where(self.velocity > max_speed_kmh)[0]
        indices = np.unique(
            np.concatenate(
                [
                    offset + high_speed_indices
                    for offset in range(-tolerance, tolerance + 1)
                ]
            )
        )
        indices = indices[(indices > 0) & (indices < len(self.velocity) - 1)]
        mask = np.ones(len(self.velocity), dtype=bool)
        mask[indices] = False
        return self.trim_mask(mask, interpolate=interpolate)

    def trim_between_datetime(
        self, start_time: datetime, end_time: datetime, interpolate=True
    ):
        mask = (self.time <= start_time) | (self.time >= end_time)
        return self.trim_mask(mask, interpolate=interpolate)

    def recalibrate_barometer_elevation(
        self, min_elevation: str | None = None, max_elevation: str | None = None
    ) -> "GpxTrack":
        """Cure barometer elevation data by removing unrealistic elevation values.

        Parameters
        ----------
        - min_elevation: minimum realistic elevation (e.g. 0 for sea level)
        - max_elevation: maximum realistic elevation (e.g. 8848 for Mount Everest)

        Returns
        -------
        - GpxTrack: a new GpxTrack instance with cured elevation data
        """
        if self.elevation is None:
            return self

        max_seen_elevation = max(self.elevation)
        min_seen_elevation = min(self.elevation)
        if max_elevation is None:
            max_elevation = max_seen_elevation
        if min_elevation is None:
            min_elevation = min_seen_elevation
        f = interp1d(
            [min_seen_elevation, max_seen_elevation], [min_elevation, max_elevation]
        )
        return GpxTrack(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=f(self.elevation),
            time=self.time,
        )

    def smooth(
        self, n=5, window="hanning", start_time=None, end_time=None, segment=None
    ):
        """Smooth position data (and subsequently distance, velocity etc.)

        Parameters
        ----------
        - n: size of moving window for smoothing
        - window: type of window (e.g. 'hanning' or 'flat', see gpxo.smooth())
        """
        # Create copies to avoid modifying original data directly
        latitude_copy = self.latitude.copy()
        longitude_copy = self.longitude.copy()
        elevation_copy = self.elevation.copy()

        # Determine the indices for the smoothing window
        start_index = 0
        end_index = len(self.time)

        if start_time:
            # Find the index of the first point at or after start_time
            start_index = next(
                (i for i, t in enumerate(self.time) if t >= start_time), 0
            )

        if end_time:
            # Find the index of the last point at or before end_time
            # We search backwards from the end to be more efficient
            end_index = (
                next(
                    (
                        i
                        for i, t in reversed(list(enumerate(self.time)))
                        if t <= end_time
                    ),
                    len(self.time) - 1,
                )
                + 1
            )

        # If the start and end indices are valid, apply smoothing
        if end_index > start_index:
            latitude_copy[start_index:end_index] = smooth(
                self.latitude[start_index:end_index], n=n, window=window
            )
            longitude_copy[start_index:end_index] = smooth(
                self.longitude[start_index:end_index], n=n, window=window
            )
            elevation_copy[start_index:end_index] = smooth(
                self.elevation[start_index:end_index], n=n, window=window
            )

        return GpxTrack(
            name=self.name,
            latitude=latitude_copy,
            longitude=longitude_copy,
            elevation=elevation_copy,
            time=self.time,
        )

    def display(self):
        """Prints a nicely formatted summary of the GPX track with a clean box layout."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        print()
        table = Table(title="GPX Summary", show_header=False, box=box.DOUBLE_EDGE)

        # --- Calculations (unchanged) ---
        if self.time is None or len(self.time) == 0:
            print("--- GPX Track Summary ---")
            print("Error: The track has no data to summarize.")
            return

        duration_s = self.seconds[-1] - self.seconds[0]
        duration_td = timedelta(seconds=int(duration_s))

        # Reformat time to be more readable
        start_time_str = self.time[0].strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = self.time[-1].strftime("%Y-%m-%d %H:%M:%S")

        # Correct elevation calculations
        elev_diff = np.diff(self.elevation)
        elev_gain = np.sum(elev_diff[elev_diff > 0])
        elev_loss = np.sum(elev_diff[elev_diff < 0])

        # Display basic stats
        table.add_row("Track name", self.name, end_section=True)
        table.add_row("Number of points", str(self.n_points))
        table.add_row("Start time", start_time_str)
        table.add_row("End time", end_time_str)
        table.add_row("Duration", str(duration_td))
        table.add_row(
            "Total distance (km)", f"{self.distance[-1]:.2f}", end_section=True
        )

        if self.velocity is not None:
            avg_speed = np.mean(self.velocity)
            max_speed = np.max(self.velocity)
            table.add_row("Average speed (km/h)", f"{avg_speed:.2f}")
            table.add_row("Max speed (km/h)", f"{max_speed:.2f}", end_section=True)

        if self.elevation is not None:
            min_elev = np.min(self.elevation)
            max_elev = np.max(self.elevation)

            table.add_row("Min elevation (m)", f"{min_elev:.2f}")
            table.add_row("Max elevation (m)", f"{max_elev:.2f}")
            table.add_row("Elevation gain (m)", f"{elev_gain:.2f}")
            table.add_row("Elevation loss (m)", f"{elev_loss:.2f}", end_section=True)

        console = Console()
        console.print(table)

    def save(self, file: str) -> None:
        """Save the GPX track to a file."""
        import gpxpy
        import gpxpy.gpx

        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack(name=self.name)
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        for lat, lon, ele, time in zip(
            self.latitude, self.longitude, self.elevation, self.time
        ):
            gpx_segment.points.append(
                gpxpy.gpx.GPXTrackPoint(
                    latitude=lat, longitude=lon, elevation=ele, time=time
                )
            )

        with open(file, "w") as f:
            f.write(gpx.to_xml())

    def __repr__(self):
        return f"GpxTrack(name='{self.name}', n_points={self.n_points}"
