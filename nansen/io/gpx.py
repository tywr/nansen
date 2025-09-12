from nansen.core.gpx_data_frame import GpxDataFrame


def load_raw_gpx(file):
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

    data = []
    tracks = set()
    routes = set()

    # # Waypoints
    # for waypoint in gpx.waypoints:
    #     data.append(
    #         {
    #             "type": "waypoint",
    #             "name": waypoint.name,
    #             "latitude": waypoint.latitude,
    #             "longitude": waypoint.longitude,
    #             "elevation": waypoint.elevation,
    #             "time": waypoint.time,
    #             "description": waypoint.description,
    #         }
    #     )

    # Tracks
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                tracks.add(track.name)
                data.append(
                    {
                        "type": "track_point",
                        "track_name": track.name,
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "elevation": point.elevation,
                        "time": point.time,
                    }
                )

    # Routes
    for route in gpx.routes:
        for point in route.points:
            routes.add(route.name)
            data.append(
                {
                    "type": "route_point",
                    "route_name": route.name,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "elevation": point.elevation,
                    "time": point.time,
                }
            )

    return tracks, routes, data


def read_gpx(
    file: str, track: str | None = None, route: str | None = None
) -> GpxDataFrame:
    """
    Read a GPX file and return the data as a nansen GpxDataFrame.

    Parameters
    ----------
    file : str
        Path to the GPX file.
    track : str, optional
        Name of the track to extract. If None, the only track in the file is extracted.
    route : str, optional
        Name of the route to extract. If None, the only route in the file is extracted.

    Returns
    -------
    nansen.GpxDataFrame
        Data frame containing waypoints, tracks, and routes in the GPX file.
    """
    import pandas as pd

    selected = None
    gpx_type = None

    if track and route:
        raise ValueError("Only one of 'track' or 'route' can be specified.")
    if track:
        gpx_type = "track"
    if route:
        gpx_type = "route"

    tracks, routes, raw = load_raw_gpx(file)

    if not track:
        if len(tracks) == 1:
            selected = tracks.pop()
            gpx_type = "track"
        elif len(tracks) > 1:
            raise ValueError(f"Multiple tracks found in file. Specify one of: {tracks}")
    elif track not in tracks:
        raise ValueError(
            f"Track '{track}' not found in file. Available tracks: {tracks}"
        )

    if not route:
        if len(routes) == 1:
            if selected:
                raise ValueError(
                    "File contains both tracks and routes. Specify only one."
                )
            else:
                selected = routes.pop()
                gpx_type = "route"
        elif len(routes) > 1:
            raise ValueError(f"Multiple routes found in file. Specify one of: {routes}")
    elif route not in routes:
        raise ValueError(
            f"Route '{route}' not found in file. Available routes: {routes}"
        )

    raw = (
        [
            {
                k: v
                for k, v in entry.items()
                if k in ["latitude", "longitude", "elevation", "time"]
            }
            for entry in raw
            if entry.get("track_name") == selected
            or entry.get("route_name") == selected
        ]
        if selected
        else raw
    )
    gpx_df = pd.DataFrame(raw)
    gpx_df["time"] = pd.to_datetime(gpx_df["time"], errors="coerce")
    return GpxDataFrame(gpx_df=gpx_df, type=gpx_type)


if __name__ == "__main__":
    gpx_df = read_gpx("/data/alpi.gpx")
    print(gpx_df)
