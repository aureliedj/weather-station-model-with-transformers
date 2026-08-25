"""Data loading and station visualisation.

Nothing is re-exported eagerly. This file is executed by any
`from data.dataset import ...`, so a re-export here would make every training
and evaluation run import the whole geospatial plotting stack (geopandas,
rioxarray, matplotlib) that only a notebook uses. Import from the module:

    from data.dataset   import load_peakweather, StationMAEDataset
    from data.visualize import plot_stations_on_dem

Same policy as engine/__init__.py and model/__init__.py.
"""
