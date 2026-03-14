"""
data/visualize.py

Visualization utilities for the Station-MAE project.

Main function:
    plot_stations_on_dem(ds, stations, path_swissshape, ...)
        Plots the Swiss DEM with optional station overlays, each with
        a custom color code and label.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data structure for station overlay points
# ---------------------------------------------------------------------------

@dataclass
class StationMarker:
    """
    Represents a station (or any point) to be plotted on the map.

    Args:
        name:       Station identifier / label shown in the legend.
        easting:    Swiss LV95 easting  (EPSG:2056, metres).
        northing:   Swiss LV95 northing (EPSG:2056, metres).
        color:      Matplotlib color string, hex code, or RGB tuple.
                    Used as the face color of the scatter marker.
        marker:     Matplotlib marker style (default "o").
        size:       Marker size in points² (default 50).
        zorder:     Drawing order — higher values appear on top (default 10).
    """
    name:     str
    easting:  float
    northing: float
    color:    str   = "red"
    marker:   str   = "o"
    size:     float = 50
    zorder:   int   = 10


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_stations_on_dem(
    ds,
    stations:        list[StationMarker],
    path_swissshape: str,
    *,
    coarsen_factor:  int            = 10,
    figsize:         tuple          = (10, 10),
    title:           str            = "Switzerland DEM with Station Overlay",
    show_legend:     bool           = True,
    legend_kwargs:   Optional[dict] = None,
    save_path:       Optional[str]  = None,
    ax:              Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the Swiss DEM with a border and an arbitrary set of station markers.

    The DEM is rendered in two layers:
      1. Full DEM at low opacity — gives geographic context outside Switzerland.
      2. Switzerland-clipped DEM at full opacity — highlights the study area.

    Args:
        ds:
            PeakWeatherDataset instance. Must support ds.load_topography()
            returning a dataset with a "topo_DEM" variable containing a "dem"
            DataArray in Swiss LV95 coordinates.
        stations:
            List of StationMarker objects. Each defines one group of points
            to overlay (name, easting, northing, color, marker style).
            Pass an empty list to plot the DEM only.
        path_swissshape:
            Path to the swissBOUNDARIES3D shapefile or zip. The layer
            "swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET" is used automatically.
        coarsen_factor:
            Spatial downsampling factor applied to the DEM before plotting.
            Higher values = faster rendering, lower resolution (default 10).
        figsize:
            Figure size in inches (default (10, 10)).
        title:
            Plot title string.
        show_legend:
            Whether to show a legend for the station markers (default True).
        legend_kwargs:
            Optional dict of keyword arguments forwarded to ax.legend().
        save_path:
            If provided, save the figure to this path instead of (or in
            addition to) displaying it. E.g. "outputs/dem_stations.png".
        ax:
            Optional existing Axes to draw into. If None, a new figure and
            axes are created.

    Returns:
        matplotlib Axes object.

    Example:
        >>> from data.visualize import plot_stations_on_dem, StationMarker
        >>> markers = [
        ...     StationMarker("KLO", easting=2682900, northing=1254100, color="red"),
        ...     StationMarker("ABO", easting=2609372, northing=1148939, color="blue"),
        ... ]
        >>> plot_stations_on_dem(ds, markers, path_swissshape="path/to/shapes.zip")
    """
    # ------------------------------------------------------------------
    # 1. Load Switzerland boundary
    # ------------------------------------------------------------------
    switzerland = gpd.read_file(
        path_swissshape,
        layer="swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET",
    ).to_crs("EPSG:2056")

    minx, miny, maxx, maxy = switzerland.total_bounds

    # ------------------------------------------------------------------
    # 2. Load and prepare DEM
    # ------------------------------------------------------------------
    topo = ds.load_topography()
    dem  = topo["topo_DEM"].dem

    # Clip to Switzerland border
    dem_ch = dem.rio.clip(
        switzerland.geometry,
        switzerland.crs,
        drop=False,
    )

    # Downsample both full and clipped DEM for faster rendering
    dem_small    = dem.coarsen(   x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()
    dem_ch_small = dem_ch.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()

    # Crop to Switzerland bounding box
    dem_bg        = dem_small.sel(   x=slice(minx, maxx), y=slice(miny, maxy))
    dem_foreground = dem_ch_small.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    # ------------------------------------------------------------------
    # 3. Create figure
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ------------------------------------------------------------------
    # 4. Plot DEM layers
    # ------------------------------------------------------------------
    # Background: full DEM, low opacity for geographic context
    dem_bg.plot(
        ax=ax,
        cmap="terrain",
        alpha=0.4,
        robust=True,
        add_labels=False,
        add_colorbar=False,
    )

    # Foreground: Switzerland-clipped DEM, full opacity with colorbar
    dem_foreground.plot(
        ax=ax,
        cmap="terrain",
        robust=True,
        add_labels=False,
        cbar_kwargs={
            "label":  "Elevation (m)",
            "shrink": 0.5,
            "pad":    0.01,
        },
    )

    # Switzerland border
    switzerland.boundary.plot(ax=ax, color="black", linewidth=0.8)

    # ------------------------------------------------------------------
    # 5. Plot station markers
    # ------------------------------------------------------------------
    for station in stations:
        ax.scatter(
            station.easting,
            station.northing,
            color=station.color,
            edgecolor="black",
            linewidths=0.5,
            s=station.size,
            marker=station.marker,
            zorder=station.zorder,
            label=station.name,
        )

    # ------------------------------------------------------------------
    # 6. Legend, title, formatting
    # ------------------------------------------------------------------
    if show_legend and stations:
        _legend_kwargs = {"framealpha": 0.8, "fontsize": 8, "markerscale": 1.2}
        if legend_kwargs:
            _legend_kwargs.update(legend_kwargs)
        ax.legend(**_legend_kwargs)

    ax.set_title(title, fontsize=12, pad=12)
    ax.axis("off")
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Save / show
    # ------------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Convenience: build StationMarker list from ds.stations_table
# ---------------------------------------------------------------------------

def markers_from_stations_table(
    stations_table,
    station_ids:    Optional[list[str]] = None,
    color_column:   Optional[str]       = None,
    default_color:  str                 = "red",
    cmap:           str                 = "tab10",
    marker:         str                 = "o",
    size:           float               = 50,
) -> list[StationMarker]:
    """
    Build a list of StationMarker objects directly from ds.stations_table.

    Args:
        stations_table:
            DataFrame from ds.stations_table. Must contain columns
            "swiss_easting" and "swiss_northing". Index should be station IDs.
        station_ids:
            Subset of station IDs to include. If None, all stations are used.
        color_column:
            Optional column in stations_table whose values are used to
            color-code the markers (e.g. a cluster label or station type).
            If None, all markers use default_color.
        default_color:
            Color used when color_column is None.
        cmap:
            Matplotlib colormap name used to map color_column values to colors
            when color_column is provided.
        marker:
            Marker style applied to all stations.
        size:
            Marker size applied to all stations.

    Returns:
        List of StationMarker objects ready to pass to plot_stations_on_dem().

    Example:
        >>> markers = markers_from_stations_table(
        ...     ds.stations_table,
        ...     color_column="station_type",
        ... )
        >>> plot_stations_on_dem(ds, markers, path_swissshape=PATH)
    """
    table = stations_table
    if station_ids is not None:
        table = table.loc[station_ids]

    # Build color map if a grouping column is provided
    if color_column is not None:
        unique_vals  = table[color_column].unique()
        colormap     = cm.get_cmap(cmap, len(unique_vals))
        color_lookup = {
            val: mcolors.to_hex(colormap(i))
            for i, val in enumerate(unique_vals)
        }

    result = []
    for station_id, row in table.iterrows():
        color = (
            color_lookup[row[color_column]]
            if color_column is not None
            else default_color
        )
        result.append(StationMarker(
            name=str(station_id),
            easting=float(row["swiss_easting"]),
            northing=float(row["swiss_northing"]),
            color=color,
            marker=marker,
            size=size,
        ))

    return result