"""
Contains utility functions for handling and enriching elevation data.
"""

import os
import time
import requests
import numpy as np
import tqdm

import nansen as ns


def get_elevation_opentopo(
    coordinates: list[tuple[float, float]], dataset: str = "test-dataset"
):
    url = "https://api.opentopodata.org/v1/"
    endpoint = os.path.join(url, dataset)
    params = {"locations": "|".join(f"{lat},{lon}" for lat, lon in coordinates)}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return results
    else:
        raise response.raise_for_status()


def get_elevations_opentopo(
    track: ns.GpxTrack,
    dataset: str = "test-dataset",
    distance_between_calls: float = 3e-2,
):
    """
    Get elevation data from OpenTopoData for a list of latitude and longitude coordinates.
    """
    # Get the incremental distance in meters for each gpx point
    mask = np.insert(
        np.diff(1 + np.floor(track.distance / distance_between_calls)), 0, 1
    ).astype(bool)
    mask[-1] = True  # Always include the last point

    latitudes = track.latitude[mask]
    longitudes = track.longitude[mask]
    print("Number of points to query:", len(latitudes))
    print("Querying opentopodata API...")
    elevations = np.zeros(latitudes.shape)
    for i in tqdm.tqdm(range(0, latitudes.size, 100)):
        coords = list(zip(latitudes[i : i + 100], longitudes[i : i + 100]))
        results = get_elevation_opentopo(coords, dataset=dataset)
        for j, result in enumerate(results):
            if "elevation" in result:
                elevations[i + j] = result["elevation"]
        time.sleep(1)
    elevations_full = np.full(track.latitude.shape, np.nan)
    elevations_full[mask] = elevations

    known_indices = np.where(~np.isnan(elevations_full))[0]
    known_values = elevations_full[known_indices]
    nan_indices = np.where(np.isnan(elevations_full))[0]
    interpolated_elevations = np.interp(
        nan_indices,
        known_indices,
        known_values,
    ).astype(np.int64)
    elevations_full[nan_indices] = interpolated_elevations
    return elevations_full
