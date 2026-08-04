"""
data — PeakWeather loading and windowing for Station-MAE v15.

Training only. The station-map plotting helpers (data/visualize.py in the full
project) are deliberately not re-exported here: they pull in geopandas,
rioxarray and matplotlib, none of which training needs.
"""

from .dataset import load_peakweather, StationMAEDataset, TRAIN_YEARS

__all__ = ["load_peakweather", "StationMAEDataset", "TRAIN_YEARS"]
