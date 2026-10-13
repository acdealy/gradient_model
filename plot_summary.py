'''
plot_summary.py

Summary figure for a model run:
    1. net change in eta by site
    2. shoreline change within each output interval (not cumulative) (time vs. site)
    3. time-mean Qs by site
    4. mean wave height by site, per partition
    5. transport-weighted wave directions, per partition
    6. annual wave transport proxy, per partition

and a second figure of the coast before and after: an overview map colored by net change, plus
zoomed panels (true scale) at the largest changes or at sites listed in ZOOM_SITES.

Wave statistics cover the same period as the model run. Uses the settings in config.py.

    python plot_summary.py                          # output file from config.py
    python plot_summary.py output/other_run.nc      # a different output file
'''

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from pyproj import Transformer

from shoreline_model import SECONDS_PER_YEAR, get_nearest_indices, smooth

# optional landmarks drawn on the by-site panels (skipped if not in the run's domain)
MARKERS = {}           # e.g. {'Rodanthe': 'usa_NC_0036_0027'}

PARTITION_COLORS = ['#023E8A', '#0077B6', '#48CAE4', '#ADE8F4']

# before/after figure: sites to zoom in on; leave empty to zoom on the largest changes
ZOOM_SITES = []        # e.g. ['usa_NC_0036_0027']
N_ZOOMS = 3


def load_waves(cfg, t_start, t_end, site_lat, site_lon):

    ''' Wave data for the run period at the wave points that force the model.
    Returns {var: array (partition, time, used wave point)}, wave times, and, for each site,
    the column of its wave point in those arrays (nearest point, as in the model). '''

    ref = xr.open_dataset(cfg.path(cfg.wave_files['dp'].format(i=cfg.partitions[0])))
    mapping = get_nearest_indices(ref['lat'].values, ref['lon'].values, site_lat, site_lon)
    used, site_col = np.unique(mapping, return_inverse=True)
    times = ref['time'].sel(time=slice(t_start, t_end)).values
    ref.close()

    waves = {}
    for var in ('dp', 'hs', 'tp'):
        arrs = []
        for p in cfg.partitions:
            ds = xr.open_dataset(cfg.path(cfg.wave_files[var].format(i=p)))
            vals = ds[var].sel(time=slice(t_start, t_end)).transpose('time', 'site').values
            arrs.append(vals[:, used].astype(float))  # keep only the wave points the model uses
            ds.close()
        waves[var] = np.stack(arrs)
    return waves, times, site_col


def plot_summary(cfg, output_file=None, png=None):

    output_file = Path(output_file or cfg.output_file)
    out = xr.open_dataset(output_file)
    site_ids = out['site'].values.astype(str)
    idx = np.arange(len(site_ids))
    times = out['time'].values
    years = (times[-1] - times[0]) / np.timedelta64(1, 'D') / 365.25

    # 1. net change: only the first and last records are needed
    net = (out['eta'].isel(time=-1) - out['eta'].isel(time=0)).values

    # 2. change within each interval (not cumulative), thinned to at most ~600 intervals
    stride = max(1, int(np.ceil(len(times) / 600)))
    sub = out['eta'].isel(time=slice(None, None, stride))
    d_eta = np.diff(sub.values, axis=0)        # eta(t) - eta(t - interval)
    sub_times = sub['time'].values[1:]         # each row is labeled by the end of its interval
    interval_days = np.median(np.diff(sub['time'].values)) / np.timedelta64(1, 'D')

    # 3. time-mean Qs, weighting each record by its interval, read in blocks to limit memory
    w = np.diff(times).astype('timedelta64[s]').astype(float)
    qs_sum = np.zeros(len(site_ids))
    for i0 in range(1, len(times), 500):
        i1 = min(i0 + 500, len(times))
        block = out['qs_mean'].isel(time=slice(i0, i1)).values
        qs_sum += np.nansum(block * w[i0 - 1:i1 - 1, None], axis=0)
    qs_time_mean = qs_sum / w.sum()

    # 4-6. waves over the run period, mapped to sites the same way as the model
    print("Loading wave data...")
    shore_sigma = float(out.attrs.get('shore_sigma', cfg.shore_sigma))
    waves, w_times, site_col = load_waves(cfg, times[0], times[-1],
                                          smooth(out['lat'].values, shore_sigma),
                                          smooth(out['lon'].values, shore_sigma))
    n_used = waves['hs'].shape[2]
    proxy = np.nan_to_num(waves['hs']**2.4 * waves['tp']**0.2)  # magnitude part of the Qs formula

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), layout='constrained')
    markers = {name: int(np.where(site_ids == s)[0][0])
               for name, s in MARKERS.items() if s in site_ids}

    def add_markers(ax):
        for name, i in markers.items():
            ax.axvline(i, color='#FF8DA1', linestyle='--', lw=1)
            ax.text(i, 1.0, f' {name}', transform=ax.get_xaxis_transform(),
                    color='#FF8DA1', va='top', fontsize=9)

    # (1) net change in eta by site
    ax = axes[0, 0]
    ax.plot(idx, net, color='#35063E', lw=1)
    ax.fill_between(idx, net, 0, where=net >= 0, color='#0077B6', alpha=0.3, label='accretion')
    ax.fill_between(idx, net, 0, where=net < 0, color='#FE6D4E', alpha=0.3, label='erosion')
    ax.axhline(0, color='k', lw=0.5)
    add_markers(ax)
    ax.set_xlabel('Index (S->N)')
    ax.set_ylabel(r'Net $\Delta \eta$ (m)')
    ax.set_title(f'Net Shoreline Change ({pd.Timestamp(times[0]):%Y-%m-%d} to '
                 f'{pd.Timestamp(times[-1]):%Y-%m-%d}, {years:.1f} yr)')
    ax.legend(loc='lower left')

    # (2) shoreline change within each interval: blue = accreting then, red = eroding then
    ax = axes[0, 1]
    lim = np.nanpercentile(np.abs(d_eta), 99.5) or 1.0
    pc = ax.pcolormesh(idx, sub_times, d_eta, cmap='RdBu', vmin=-lim, vmax=lim, shading='auto')
    add_markers(ax)
    ax.set_xlabel('Index (S->N)')
    ax.set_ylabel('Time')
    ax.set_title(f'Shoreline Change per {interval_days:.0f}-Day Interval')
    plt.colorbar(pc, ax=ax, label=rf'$\Delta \eta$ per {interval_days:.0f} days (m)',
                 fraction=0.03, pad=0.02)

    # (3) time-mean Qs by site
    ax = axes[1, 0]
    ax.plot(idx, qs_time_mean * SECONDS_PER_YEAR / 1e3, color='#023E8A', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    add_markers(ax)
    ax.set_xlabel('Index (S->N)')
    ax.set_ylabel(r'Mean $Q_s$ (10$^3$ m$^3$/yr)')
    ax.set_title(r'Time-Mean $Q_s$ (positive = southward / toward lower index)')

    # (4) mean wave height by site, per partition (days the partition is present)
    ax = axes[1, 1]
    for j, p in enumerate(cfg.partitions):
        hs_site = np.nanmean(waves['hs'][j], axis=0)[site_col]
        present = np.mean(np.isfinite(waves['hs'][j]))
        ax.plot(idx, hs_site, color=PARTITION_COLORS[j], lw=1.5,
                label=f'partition {p} (present {present:.0%} of days)')
    add_markers(ax)
    ax.set_xlabel('Index (S->N)')
    ax.set_ylabel(r'Mean $H_s$ when present (m)')
    ax.set_title('Mean Wave Height by Site')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom so the legend doesn't cover the lines
    ax.legend(loc='upper left')

    # (5) transport-weighted direction histogram over all wave points that force the model
    ax = axes[2, 0]
    bins = np.arange(0, 361, 10)
    bottom = np.zeros(len(bins) - 1)
    for j, p in enumerate(cfg.partitions):
        dp = waves['dp'][j].ravel()
        wt = proxy[j].ravel()
        ok = np.isfinite(dp)
        h, _ = np.histogram(dp[ok] % 360, bins=bins, weights=wt[ok])
        h = h / n_used / years  # per wave point per year
        ax.bar(bins[:-1], h, width=10, bottom=bottom, align='edge',
               color=PARTITION_COLORS[j], label=f'partition {p}')
        bottom += h
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel('Peak direction, coming from (deg)')
    ax.set_ylabel(r'$\sum H_s^{2.4} T_p^{0.2}$ per wave point per year')
    ax.set_title('Transport-Weighted Wave Directions')
    ax.legend()

    # (6) annual transport proxy, per partition (absent partitions count as 0)
    ax = axes[2, 1]
    yr = pd.DatetimeIndex(w_times).year
    annual = pd.DataFrame({f'partition {p}': proxy[j].mean(axis=1)
                           for j, p in enumerate(cfg.partitions)}, index=yr).groupby(level=0).mean()
    ax.stackplot(annual.index, annual.T.values, labels=annual.columns, colors=PARTITION_COLORS)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Mean daily $H_s^{2.4} T_p^{0.2}$')
    ax.set_title('Annual Wave Transport Proxy by Partition')
    ax.legend(loc='upper left')

    fig.suptitle(f"Shoreline Model Summary: $D_{{eff}}$ = {out.attrs.get('d_eff', '?')} m, "
                 f"$\\gamma$ = {out.attrs.get('gamma', '?')} pts, "
                 f"s_x = {Path(str(out.attrs.get('s_x_file', 'None'))).name}", fontsize=16)

    png = Path(png or output_file.with_name(output_file.stem + '_summary.png'))
    fig.savefig(png, dpi=110)
    out.close()
    print(f"Saved: {png}")
    return png


def load_shorelines(cfg, out):

    ''' Initial and final shoreline lat/lon. Uses the saved shore_lat/shore_lon, or rebuilds
    them from eta and the transects if the run was saved without them. '''

    if 'shore_lat' in out:
        return (out['shore_lat'].isel(time=0).values, out['shore_lon'].isel(time=0).values,
                out['shore_lat'].isel(time=-1).values, out['shore_lon'].isel(time=-1).values)

    import geopandas as gpd
    from shoreline_model import TransectGeometry
    gdf = gpd.read_file(cfg.path(cfg.transects_file)).to_crs(cfg.utm_crs).set_index('id')
    geom = TransectGeometry(gdf, list(out['site'].values.astype(str)), cfg.utm_crs)
    lat0, lon0 = geom.eta_to_latlon(out['eta'].isel(time=0).values)
    lat1, lon1 = geom.eta_to_latlon(out['eta'].isel(time=-1).values)
    return lat0, lon0, lat1, lon1


def plot_coast_change(cfg, output_file=None, png=None):

    output_file = Path(output_file or cfg.output_file)
    out = xr.open_dataset(output_file)
    site_ids = out['site'].values.astype(str)
    times = out['time'].values
    net = (out['eta'].isel(time=-1) - out['eta'].isel(time=0)).values
    lat0, lon0, lat1, lon1 = load_shorelines(cfg, out)

    # work in meters for the zooms
    to_utm = Transformer.from_crs('EPSG:4326', cfg.utm_crs, always_xy=True)
    x0, y0 = to_utm.transform(lon0, lat0)
    x1, y1 = to_utm.transform(lon1, lat1)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x0), np.diff(y0)))])

    # each window is sized so the change is visible at true scale (500 m to 10 km half-width)
    def half_width(i):
        return float(np.clip(15 * abs(net[i]), 500, 10e3))

    # zoom centers: requested sites, or the largest accretion, the largest erosion, then the
    # next-largest changes, keeping windows from overlapping
    if ZOOM_SITES:
        centers = [int(np.where(site_ids == z)[0][0]) for z in ZOOM_SITES if z in site_ids]
    else:
        net_f = np.nan_to_num(net)
        order = [int(np.argmax(net_f)), int(np.argmin(net_f))] + list(np.argsort(-np.abs(net_f)))
        centers = []
        for i in order:
            if all(abs(s[i] - s[c]) > half_width(i) + half_width(c) for c in centers):
                centers.append(int(i))
            if len(centers) == N_ZOOMS:
                break

    windows = [(c, half_width(c), np.abs(s - s[c]) <= half_width(c)) for c in centers]

    n = max(len(windows), 1)
    fig = plt.figure(figsize=(17, 5.2 * n), layout='constrained')
    gs = fig.add_gridspec(n, 2, width_ratios=[1.1, 1])
    lim = np.nanpercentile(np.abs(net), 99.5) or 1.0
    letters = 'ABCDEFGH'

    # overview map
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(lon0, lat0, color='0.75', lw=4, label='initial shoreline', zorder=1)
    sc = ax.scatter(lon1, lat1, c=net, cmap='RdBu', vmin=-lim, vmax=lim, s=4, zorder=2,
                    label='final shoreline')
    for k, (c, half, sel) in enumerate(windows):
        pad = 0.01
        lo, la = np.r_[lon0[sel], lon1[sel]], np.r_[lat0[sel], lat1[sel]]
        ax.add_patch(Rectangle((lo.min() - pad, la.min() - pad), np.ptp(lo) + 2 * pad,
                               np.ptp(la) + 2 * pad, fill=False, ec='k', lw=1.2, zorder=3))
        ax.text(lo.max() + 2 * pad, la.mean(), letters[k], fontsize=14, weight='bold', va='center')
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lat0))))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Shoreline {pd.Timestamp(times[0]):%Y-%m-%d} to {pd.Timestamp(times[-1]):%Y-%m-%d}')
    ax.legend(loc='lower right')
    plt.colorbar(sc, ax=ax, label=r'Net $\Delta \eta$ (m)', fraction=0.04, pad=0.02)

    # zoomed panels at true scale, in km relative to the window center
    for k, (c, half, sel) in enumerate(windows):
        ax = fig.add_subplot(gs[k, 1])
        xc, yc = x0[c], y0[c]
        segs = np.stack([np.c_[(x0[sel] - xc) / 1e3, (y0[sel] - yc) / 1e3],
                         np.c_[(x1[sel] - xc) / 1e3, (y1[sel] - yc) / 1e3]], axis=1)
        colors = np.where(net[sel] >= 0, '#0077B6', '#FE6D4E')
        ax.add_collection(LineCollection(segs, colors=colors, lw=1, alpha=0.6))
        ax.plot((x0[sel] - xc) / 1e3, (y0[sel] - yc) / 1e3, color='0.5', ls='--', lw=1.5, label='initial')
        ax.plot((x1[sel] - xc) / 1e3, (y1[sel] - yc) / 1e3, color='k', lw=1.5, label='final')
        h = 1.05 * half / 1e3
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting from center (km)')
        ax.set_ylabel('Northing from center (km)')
        ax.set_title(f'{letters[k]}: around {site_ids[c]} | net change at center {net[c]:+.1f} m, '
                     f'range {np.nanmin(net[sel]):+.1f} to {np.nanmax(net[sel]):+.1f} m', fontsize=11)
        if k == 0:
            ax.plot([], [], color='#0077B6', label='accretion')
            ax.plot([], [], color='#FE6D4E', label='erosion')
            ax.legend(loc='best', fontsize=9)

    fig.suptitle('Coast Before and After (zooms at true scale)', fontsize=16)
    png = Path(png or output_file.with_name(output_file.stem + '_coast.png'))
    fig.savefig(png, dpi=110)
    out.close()
    print(f"Saved: {png}")
    return png


if __name__ == '__main__':
    from config import CONFIG
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    plot_summary(CONFIG, output_file=output_file)
    plot_coast_change(CONFIG, output_file=output_file)
