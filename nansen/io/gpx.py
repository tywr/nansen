from nansen.core.gpx import GpxTrack


def load_raw_gpx_tracks(file) -> list[dict]:
    """
    Load a GPX file and return the data as a list of dictionaries.

    Parameters
    ----------
    file : str
        Path to the GPX file.

    Returns
    -------
    tracks : set
        Set of track names in the GPX file.
    routes : set
        Set of route names in the GPX file.
    list of dict
        List of waypoints, tracks, and routes in the GPX file.
    """
    import gpxpy

    with open(file, "r") as f:
        gpx = gpxpy.parse(f)

    tracks = []
    for i, track in enumerate(gpx.tracks):
        for j, segment in enumerate(track.segments):
            tracks.append(
                {
                    "track_name": track.name,
                    "track_index": i,
                    "points": [
                        {
                            "segment_index": j,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                            "elevation": point.elevation,
                            "time": point.time,
                        }
                        for point in segment.points
                    ],
                }
            )

    return tracks


def read_gpx_track(
    file: str, track_index: int | None = None, track_name: str | None = None
) -> GpxTrack:
    """
    Read a GPX file and return the data as a nansen GpxDataFrame.

    Parameters
    ----------
    file : str
        Path to the GPX file.
    track_index : int, optional
        Index of the track to read, by default 0
    track_name : str, optional
        Name of the track to read, by default None

    Returns
    -------
    nansen.GpxDataFrame
        Data frame containing waypoints, tracks, and routes in the GPX file.
    """
    import pandas as pd

    if track_index is not None and track_name is not None:
        raise ValueError("Only one of track_index or track_name can be specified.")
    elif track_index is None and track_name is None:
        track_index = 0

    raw = load_raw_gpx_tracks(file)

    # Find track index by name if specified
    if track_name is not None:
        lookup = {t["track_name"]: t["track_index"] for t in raw}
        if track_name not in lookup:
            raise ValueError(f"Track name '{track_name}' not found in GPX file.")
        track_index = lookup[track_name]

    track_data = next((t for t in raw if t["track_index"] == track_index), None)
    return GpxTrack.from_points(
        name=track_data["track_name"], points=track_data["points"]
    )
