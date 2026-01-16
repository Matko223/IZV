#!/usr/bin/python3.13
# coding=utf-8

"""
@file geo.py
@brief: Analysis and visualization of collision data
@author: Martin Valapka - xvalapm00
"""

# %%%
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import KMeans

def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    @brief: Create a GeoDataFrame from accidents and locations DataFrames
    @param df_accidents: DataFrame containing accident data with 'p1' as index
    @param df_locations: DataFrame containing location data with 'p1' as index
    @return: GeoDataFrame with geometry column created from 'd' and 'e'
    """

    df_acc = df_accidents.copy()
    df_loc = df_locations.copy()

    if df_acc.index.name == "p1":
        df_acc = df_acc.reset_index()
    if df_loc.index.name == "p1":
        df_loc = df_loc.reset_index()

    df = df_acc.merge(df_loc[["p1", 'd', 'e']], on="p1", how="left")

    df['d'] = pd.to_numeric(df['d'], errors="coerce")
    df['e'] = pd.to_numeric(df['e'], errors="coerce")

    df = df.dropna(subset=['d', 'e'])
    df = df[(df['d'] != 0) & (df['e'] != 0)]

    swapped = df['d'] < df['e']
    if swapped.any():
        df.loc[swapped, ['d', 'e']] = df.loc[swapped, ['e', 'd']].values

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs="EPSG:5514")
    return gdf
    
def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    @brief: Plot wildlife collisions for specified years and region
    @param gdf: GeoDataFrame containing accident data with geometry
    @param fig_location: File path to save the figure
    @param show_figure: Boolean to indicate whether to display the figure
    """    

    df_geo = gdf.copy()
    years = [2023, 2024]
    region_code = 6
    region_name = "JHM"

    df_geo["year"] = pd.to_datetime(df_geo["date"], errors="coerce").dt.year
    df_geo = df_geo[df_geo["year"].isin(years)]
    
    df_geo["p4a_num"] = pd.to_numeric(df_geo["p4a"], errors="coerce").astype("Int64")
    df_geo = df_geo[df_geo["p4a_num"] == region_code]
    df_geo = df_geo[df_geo["p10"] == 4]

    df_geo_web = df_geo.to_crs(epsg=3857)
    
    # Size and padding for shared axes limits
    if not df_geo_web.empty:
        xmin, ymin, xmax, ymax = df_geo_web.total_bounds
        if xmax > xmin:
            xpad = (xmax - xmin) * 0.08
        else:
            xpad = 1000
        if ymax > ymin:
            ypad = (ymax - ymin) * 0.08
        else:
            ypad = 1000
        shared_xlim = (xmin - xpad, xmax + xpad)
        shared_ylim = (ymin - ypad, ymax + ypad)
    else:
        shared_xlim = None
        shared_ylim = None
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for i in range(len(years)):
        ax = axes[i]
        yr = years[i]
        sub = df_geo[df_geo["year"] == yr]

        gsub = geopandas.GeoDataFrame(sub, geometry=sub.geometry)
        gsub_web = gsub.to_crs(epsg=3857)

        gsub_web.plot(ax=ax, markersize=20, color="red", alpha=0.6, edgecolor='k', linewidth=0.2)

        if shared_xlim and shared_ylim:
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)

        contextily.add_basemap(ax, crs=gsub_web.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)

        ax.set_title(f"{region_name} ({yr}) - {len(gsub_web)}")
        ax.axis("off")

    plt.tight_layout()
    if fig_location:
        fig.savefig(fig_location, dpi=150, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(fig)

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    @brief: Plot clusters of alcohol-related collisions for specified years and region
    @param gdf: GeoDataFrame containing accident data with geometry
    @param fig_location: File path to save the figure
    @param show_figure: Boolean to indicate whether to display the figure
    """

    df_geo = gdf.copy()
    years = [2023, 2024]
    region_code = 6
    region_name = "JHM"

    df_geo["year"] = pd.to_datetime(df_geo["date"], errors="coerce").dt.year
    df_geo = df_geo[df_geo["year"].isin(years)]

    df_geo["p4a_num"] = pd.to_numeric(df_geo["p4a"], errors="coerce").astype("Int64")
    df_geo = df_geo[df_geo["p4a_num"] == region_code]
    df_geo["p11_num"] = pd.to_numeric(df_geo["p11"], errors="coerce").astype("Int64")
    df_geo = df_geo[df_geo["p11_num"] >= 4]

    df_geo_web = df_geo.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Size and padding for shared axes limits
    xmin, ymin, xmax, ymax = df_geo_web.total_bounds
    if xmax > xmin:
        xpad = (xmax - xmin) * 0.08
    else:
        xpad = 1000
    if ymax > ymin:
        ypad = (ymax - ymin) * 0.08
    else:
        ypad = 1000
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_title(f"{region_name} with alcohol related collisions")
    ax.axis("off")
    
    contextily.add_basemap(ax, crs=df_geo_web.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)

    """
    First I tried using AgglomerativeClustering, but the results were not satisfactory.
    I couldn't get well-defined clusters and the convex hulls often overlapped.
    I also couldn't find the optimal distance threshold for clustering.
    There were either too many small clusters or too few large ones.

    https://www.geeksforgeeks.org/data-science/choosing-the-right-clustering-algorithm-for-your-dataset/#overview-of-common-clustering-algorithms

    On this website I found about Kmeans clustering, which I then implemented.
    It doesn't require setting a distance threshold and produces more distinct clusters.
.
    The number of clusters is determined based on the number of points
    using the formula: n_clusters = max(3, int(sqrt(N / 10))) where N is the number of points. 
    This ensures a reasonable number of clusters.

    Invalid values in dataset
    """
    coords = np.array([[point.x, point.y] for point in df_geo_web.geometry])
    n_clusters = max(3, int(np.sqrt(len(coords) / 10)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_geo_web["cluster"] = kmeans.fit_predict(coords)
    
    cluster_counts = df_geo_web["cluster"].value_counts().sort_index()

    cmap = plt.colormaps.get_cmap("RdYlGn_r")
    norm = plt.Normalize(vmin=0, vmax=cluster_counts.max())

    # Draw convex hulls for each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = df_geo_web["cluster"] == cluster_id
        cluster_points = df_geo_web[cluster_mask]
        if len(cluster_points) >= 3:
            coords_cluster = np.array([[p.x, p.y] for p in cluster_points.geometry])
            hull = ConvexHull(coords_cluster)
            hull_points = coords_cluster[hull.vertices]
            poly = Polygon(hull_points)
            x, y = poly.exterior.xy
            ax.fill(x, y, color="gray", alpha=0.3, edgecolor="black", linewidth=1)

    df_geo_web["count"] = df_geo_web["cluster"].map(cluster_counts)
    df_geo_web.plot(ax=ax, column="count", cmap=cmap, norm=norm, markersize=20, alpha=0.8, edgecolor="k", linewidth=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", label="Number of collisions per cluster", pad=0.04, shrink=1.0)
    
    ticks = list(cbar.get_ticks())
    ticks.insert(0, 0)
    cbar.set_ticks(ticks)

    plt.tight_layout()
    if fig_location:
        fig.savefig(fig_location, dpi=150, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(fig)
    

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    gdf = make_geo(df_accidents, df_locations)

    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

    # testovani splneni zadani
    import os
    assert os.path.exists("geo1.png")
    assert os.path.exists("geo2.png")
# %%
