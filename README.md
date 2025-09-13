# NANSEN 🧭

Load, analyze and clean GPX tracks in python !


## Example

```python
import nansen as ns
import matplotlib.pyplot as plt

track = ns.read_gpx_track("data/alpi.gpx")

# Recalibrate wrong barometer by setting min and max values
track = track.recalibrate_barometer_elevation(
    min_elevation=2510, max_elevation=2797
)

# Remove some wrong data points and interpolate them
track = track.trim_between_datetime(
    start_time=datetime(2025, 9, 3, 14, 29, 45),
    end_time=datetime(2025, 9, 3, 14, 31, 0),
    interpolate=True
)

# Remove some data points with obviously wrong speed
track = track.trim_speed(max_speed_kmh=7, interpolate=True, tolerance=20)

# Smooth part of the track due to noise
track = track.smooth(
    n=100,
    start_time=datetime(2025, 9, 3, 13, 13, 0),
    end_time=datetime(2025, 9, 3, 15, 30, 0),
)

# Display a summary
track.display()

track.plot("dz")
track.plot_gpx()
plt.show()

# Save cleaned data
track.save("data/alpi_cleaned.gpx")
```
