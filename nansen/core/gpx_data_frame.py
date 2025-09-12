import pandas as pd


class GpxDataFrame:
    """
    A class to represent a GPX data frame.
    """

    def __init__(self, gpx_df: pd.DataFrame, type: str = "track"):
        self.gpx_df = gpx_df
        self.type = type

    def df_summary(self):
        return self.gpx_df.describe(include="all")

    @property
    def n_points(self):
        return len(self.gpx_df)

    def __repr__(self):
        return f"GpxDataFrame(type={self.type}, n_points={self.n_points})"
