"""
data/visualize.py

Visualization utilities for the Station-MAE project.

Main function:
    plot_stations_on_dem(ds, stations, path_swissshape, ...)
        Plots the Swiss DEM with optional station overlays, each with
        a custom color code. A legend is shown for any group that has a label.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import geopandas as gpd
import rioxarray  # noqa: F401 — required to register the .rio accessor on DataArray

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure for station overlay points
# ---------------------------------------------------------------------------

@dataclass
class StationMarker:
    """
    Represents a station (or any point) to be plotted on the map.

    Args:
        name:     Station identifier (used internally, not displayed on the map).
        easting:  Swiss LV95 easting  (EPSG:2056, metres).
        northing: Swiss LV95 northing (EPSG:2056, metres).
        color:    Matplotlib color string, hex code, or RGB tuple.
        marker:   Matplotlib marker style (default "o").
        size:     Marker size in points² (default 50).
        zorder:   Drawing order — higher values appear on top (default 10).
        label:    Legend label for this group. If None, group is not shown
                  in the legend. All markers in the same group should share
                  the same label (set via markers_from_stations_table).
    """
    name:     str
    easting:  float
    northing: float
    color:    str           = "red"
    marker:   str           = "o"
    size:     float         = 50
    zorder:   int           = 10
    label:    Optional[str] = None


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_stations_on_dem(
    ds,
    stations:        list[StationMarker],
    path_swissshape: str,
    *,
    coarsen_factor:  int                = 10,
    figsize:         tuple              = (10, 10),
    title:           str                = "Switzerland DEM with Station Overlay",
    save_path:       Optional[str]      = None,
    ax:              Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the Swiss DEM with a border and an arbitrary set of station markers.

    The DEM is rendered in two layers:
      1. Full DEM at low opacity — gives geographic context outside Switzerland.
      2. Switzerland-clipped DEM at full opacity — highlights the study area.

    A legend is automatically built from any marker group that has a label
    set (via the `label` argument of markers_from_stations_table). Groups
    without a label are plotted silently.

    Args:
        ds:
            PeakWeatherDataset instance. Must support ds.load_topography()
            returning a dataset with a "topo_DEM" variable containing a "dem"
            DataArray in Swiss LV95 coordinates.
        stations:
            List of StationMarker objects. Pass an empty list to plot DEM only.
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
        save_path:
            If provided, save the figure to this path.
            E.g. "outputs/dem_stations.png".
        ax:
            Optional existing Axes to draw into. If None, a new figure and
            axes are created.

    Returns:
        matplotlib Axes object.
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
    dem_small    = dem.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()
    dem_ch_small = dem_ch.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()

    # Crop to Switzerland bounding box
    dem_bg         = dem_small.sel(   x=slice(minx, maxx), y=slice(miny, maxy))
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
    dem_bg.plot(
        ax=ax,
        cmap="terrain",
        norm=mcolors.Normalize(vmin=0, vmax=5000),
        alpha=0.4,
        robust=True,
        add_labels=False,
        add_colorbar=False,
    )

    dem_foreground.plot(
        ax=ax,
        cmap="terrain",
        norm=mcolors.Normalize(vmin=0, vmax=5000),
        robust=True,
        add_labels=False,
        add_colorbar=False,
    )

    switzerland.boundary.plot(ax=ax, color="black", linewidth=0.8)

    # ------------------------------------------------------------------
    # 5. Plot station markers
    #    Track one representative handle per labeled group for the legend.
    # ------------------------------------------------------------------
    legend_handles = {}   # label → Line2D handle, preserves insertion order

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
        )

        # Register one legend entry per unique label
        if station.label is not None and station.label not in legend_handles:
            legend_handles[station.label] = mlines.Line2D(
                [], [],
                color=station.color,
                marker=station.marker,
                linestyle="None",
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=station.label,
            )

    # ------------------------------------------------------------------
    # 6. Legend (only if at least one group has a label)
    # ------------------------------------------------------------------
    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            framealpha=0.8,
            fontsize=8,
            markerscale=1.2,
            loc="upper left",
        )

    # ------------------------------------------------------------------
    # 7. Title and formatting
    # ------------------------------------------------------------------
    ax.set_title(title, fontsize=12, pad=12)
    ax.axis("off")
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 8. Save if requested
    # ------------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Convenience: build StationMarker list from ds.stations_table
# ---------------------------------------------------------------------------

def markers_from_stations_table(
    stations_table,
    station_ids:   Optional[list[str] | str] = None,
    color_column:  Optional[str]             = None,
    default_color: str                       = "steelblue",
    cmap:          str                       = "tab10",
    marker:        str                       = "o",
    size:          float                     = 50,
    zorder:        int                       = 10,
    label:         Optional[str]             = None,
) -> list[StationMarker]:
    """
    Build a list of StationMarker objects directly from ds.stations_table.

    Multiple calls can be combined to overlay different groups of stations
    with distinct styles. Lists are simply concatenated:

        all_markers  = markers_from_stations_table(
                           ds.stations_table,
                           label="All stations",
                       )
        highlighted  = markers_from_stations_table(
                           ds.stations_table,
                           station_ids=["PFA", "ABO"],
                           default_color="red",
                           marker="*",
                           size=200,
                           zorder=20,
                           label="Highlighted",
                       )
        plot_stations_on_dem(ds, all_markers + highlighted, ...)

    Args:
        stations_table:
            DataFrame from ds.stations_table. Must contain columns
            "swiss_easting" and "swiss_northing". Index should be station IDs.
        station_ids:
            Single station ID string ("PFA") or list (["PFA", "ABO"]) to
            include. If None, all stations in stations_table are used.
        color_column:
            Optional column in stations_table whose values are used to
            color-code markers (e.g. a cluster label or station type).
            If None, all markers use default_color.
        default_color:
            Color used when color_column is None (default "steelblue").
        cmap:
            Matplotlib colormap name used when color_column is provided.
        marker:
            Marker style applied to all stations in this group.
        size:
            Marker size (points²) applied to all stations in this group.
        zorder:
            Drawing order for this group (default 10). Use a higher value
            (e.g. 20) to ensure this group renders on top of others.
        label:
            Legend label for this group. If None (default), the group is
            plotted without a legend entry. All markers in the group share
            the same label, color and marker style in the legend.

    Returns:
        List of StationMarker objects ready to pass to plot_stations_on_dem().
    """
    table = stations_table

    # Accept both a single string and a list of strings
    if station_ids is not None:
        if isinstance(station_ids, str):
            station_ids = [station_ids]
        table = table.loc[station_ids]

    # Build color lookup if a grouping column is provided
    color_lookup = {}
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
            zorder=zorder,
            label=label,
        ))

    return result