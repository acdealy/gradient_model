'''
shoreline_model.py

One-line shoreline evolution model for the North Carolina coast.

Each wave time step:
    1. compute Qs from the partitioned wave data (dp, hs, tp) at that time and the
       shoreline lat/lon from the previous step
    2. smooth Qs with a Gaussian kernel (width gamma) and take its along-shore gradient
    3. deta/dt = -(1/D_eff) * dQ/dx + s_x
    4. eta_n = eta_{n-1} + deta/dt * dt
    5. convert eta (distance along each transect) back to lat/lon shore points

Sign convention (checked by the diffusion test in test_model.py):
    - sites are ordered S->N along the coast (increasing site index = +x)
    - calc_qs returns Qs positive for rightward transport when facing offshore, i.e. toward
      DECREASING site index. In the deta/dt formula, Q must be positive toward +x, so Q_x = -Qs.
    - get_delta_qs computes qs[i-1] - qs[i+1], which is also reversed, so
          grad = get_delta_qs(qs) / get_dists(lat, lon) = -dQs/dx = dQ_x/dx
      and the update is deta/dt = -grad / D_eff + s_x.

Omitted sites: a site whose eta changes by more than max_step_change in one wave step (or becomes
non-finite) is removed from the run instead of stopping it. The step is redone without it, it is
left out of every later calculation (its two neighbors simply become neighbors of each other), and
it is NaN in the output from that time on. If more than max_omitted_fraction of the sites end up
omitted, the run stops, since that points to numerical instability rather than a few bad points.

Units: Qs in m^3/s (as in the earlier workflow, where hourly Qs was multiplied by 3600 s),
dt in seconds, eta in meters, s_x given in m/yr and converted to m/s internally.
'''

import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
import netCDF4 as nc
import scipy.constants
from scipy.ndimage import gaussian_filter1d
from pyproj import Transformer

SECONDS_PER_YEAR = 365.25 * 86400
EARTH_RADIUS = 6371000.0  # meters


# ======================================================================================
# configuration
# ======================================================================================

@dataclass
class ModelConfig:

    ''' All model settings. Relative file names are resolved against data_dir. '''

    # --- input files (naming follows calculate_qs_partitions.ipynb and beach_w_to_gps.ipynb)
    data_dir: Path = Path('data')
    wave_files: dict = field(default_factory=lambda: {
        'dp': 'dp{i}_NorthCarolina.nc',
        'hs': 'phs{i}_NorthCarolina.nc',
        'tp': 'ptp{i}_NorthCarolina.nc',
    })
    partitions: tuple = (0, 1, 2, 3)
    transects_file: str = 'NC_transects.geojson'
    eta_init_dir: str = 'NC_CoastSat_Shorelines'   # one {transect_id}.csv per transect: time, beach_width
    s_x_file: str | None = None                     # netcdf with s_x indexed by site; None = no sources/sinks
    s_x_var: str = 's_x'
    s_x_units: str = 'm/yr'                         # 'm/yr' or 'm/s'

    # --- output
    output_file: Path = Path('shoreline_model_output.nc')
    output_every: int = 24          # write state every N wave steps (24 = daily for hourly waves)
    save_latlon: bool = True        # also save evolving shore lat/lon (time, site)
    save_diagnostics: bool = True   # save Qs and deta/dt averaged over each output interval
    restart: bool = False           # resume from the last state in output_file

    # --- time
    start_time: str | None = None   # e.g. '1984-01-01'; None = first wave time
    end_time: str | None = None     # None = last wave time
    substeps: int = 1               # split each wave step into this many model steps (same forcing)
    wave_block: int = 500           # wave time steps read from disk at once

    # --- domain (same cropping convention as the nobx analysis)
    domain_start_site: str | None = None    # e.g. 'usa_NC_0032_0001'
    domain_start_offset: int = 0            # e.g. 200
    domain_end_site: str | None = None

    # --- physics
    d_eff: float = 10.0             # effective shoreface depth (m) -- placeholder until calibrated
    gamma: float = 50.0             # Gaussian smoothing width for Qs, in number of points
    shore_sigma: float = 50.0       # Gaussian smoothing width for shoreline lat/lon before angles (points); 0 = none
    dist_sigma: float = 0.0         # smoothing width for the shoreline used in the gradient's distances (points).
                                    # 0 conserves sediment exactly; a few points removes jitter in noisy positions.
                                    # Large values (like shore_sigma) cut corners and add sediment at capes.
    k: float | None = None          # transport coefficient; None = calc_alongshore_transport_k()

    # --- initial eta from CoastSat
    eta_init_method: str = 'median'     # 'median' of obs within the window, or 'nearest' obs to start time
    eta_init_window_days: float = 365.0 # total window width centered on the start time

    # --- misc
    utm_crs: str = 'EPSG:32618'     # UTM 18N, as in beach_w_to_gps
    max_step_change: float = 10.0   # a site that moves more than this (m) in one wave step is omitted from the run
    max_omitted_fraction: float = 0.05  # stop the run if more than this fraction of sites has been omitted
                                    # (a sign of numerical instability rather than a few bad points); 1.0 never stops
    progress_every: int = 1000      # print progress every N wave steps

    def path(self, name):
        ''' Resolves a file name against data_dir unless it is already absolute. '''
        p = Path(name)
        return p if p.is_absolute() else Path(self.data_dir) / p


# ======================================================================================
# reference implementations (verbatim from the notebooks)
# these are not used in the time loop; test_model.py checks the fast versions against them
# ======================================================================================

def calc_alongshore_transport_k(gravity=scipy.constants.g, n=1.0, rho_water=1050.0, gamma_b=0.78, ):
    return (
        5.3e-6
        # * 0.46
        * rho_water
        * gravity**1.5
        * (1 / (2 * n)) ** 1.2
        * (np.sqrt(gravity * gamma_b) / (2 * np.pi)) ** 0.2
    )


def get_angles_xr(x,y):

    ''' Takes average latitude and longitude calculated in beahc_w_to_gps.ipynb and computes bearing using bearing formula
    \\(\\theta =\\>\\mathrm{atan2}\\>(\\sin \\Delta \\lambda \\cdot \\cos \\phi _{2},\\cos \\phi _{1}\\cdot \\sin \\phi _{2}-\\sin \\phi _{1}\\cdot \\cos \\phi _{2}\\cdot \\cos \\Delta \\lambda )\\)'''

    lat = np.radians(y)
    lon = np.radians(x)
    lat1 = lat.shift(site=1)
    lon1 = lon.shift(site=1)
    lat2 = lat.shift(site=-1)
    lon2 = lon.shift(site=-1)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    bearing = (bearing + 360) % 360
    return bearing.fillna(0)


def calc_qs(dp, hs, tp, angles, k):
    '''Calculates Q_s using Ashton & Murray 2006B eq. (7). Negative sign makes positive Q_s
    correspond to rightward transport when facing offshore (here, roughly southbound).'''

    tp_da = tp['tp'] if isinstance(tp, xr.Dataset) else tp
    hs_da = hs['hs'] if isinstance(hs, xr.Dataset) else hs
    dp_da = dp['dp'] if isinstance(dp, xr.Dataset) else dp

    phi  = dp_da - 90
    thet = ((angles + 90) % 360) - 90
    angle_term = ((phi - thet + 180) % 360) - 180          # wrap to [-180, 180)
    angle_term = angle_term.where(np.abs(angle_term) <= 90) # NaN where offshore-directed
    angle_term = angle_term * (np.pi / 180)

    cos_term = np.cos(angle_term)
    sin_term = np.sin(angle_term)
    qs = -k * (tp_da ** 0.2) * (hs_da ** 2.4) * (cos_term ** 1.2) * sin_term
    # qs = -k * (hs_da ** 2) * cos_term * sin_term # breaking wave formula
    return qs, angle_term


def get_dists_xr(ds):

    ''' Computes distance between adjacent points '''

    lat = np.radians(ds['lat'])
    lon = np.radians(ds['lon'])

    lat1 = lat.shift(site=1)
    lon1 = lon.shift(site=1)
    lat2 = lat.shift(site=-1)
    lon2 = lon.shift(site=-1)

    R = 6371000.0 # earth radius in meters

    phi1 = lat1
    phi2 = lat2
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1

    # haversine calculation
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance


def get_delta_qs(qs):

    ''' Calculates change in Qs between points '''

    qs1 = qs.shift(site=1)
    qs2 = qs.shift(site=-1)

    dqs = qs1-qs2

    return dqs


def get_nearest_indices(w_lats, w_lons, t_lats, t_lons):
    indices = []
    for lat, lon in zip(t_lats, t_lons):
        dist = np.sqrt((w_lats - lat)**2 + (w_lons - lon)**2)
        indices.append(np.argmin(dist))
    return np.array(indices)


# ======================================================================================
# fast numpy kernels used in the time loop (same math as the reference functions)
# ======================================================================================

def get_angles_np(lon, lat, end_fill='nearest'):

    ''' Numpy version of get_angles_xr: bearing (deg from north) from point i-1 to point i+1.
    end_fill='zero' reproduces get_angles_xr exactly (bearing 0 at both ends).
    end_fill='nearest' copies the neighboring bearing to the end points instead, so the
    domain ends don't get a spurious north-facing shoreline. '''

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat1, lat2 = lat_r[:-2], lat_r[2:]
    dlon = lon_r[2:] - lon_r[:-2]

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.empty_like(lat_r)
    bearing[1:-1] = (np.degrees(np.arctan2(y, x)) + 360) % 360
    if end_fill == 'zero':
        bearing[[0, -1]] = 0.0
    else:
        bearing[0], bearing[-1] = bearing[1], bearing[-2]
    return bearing


def calc_qs_np(dp, hs, tp, angles, k):

    ''' Numpy version of calc_qs (Ashton & Murray 2006B eq. 7). Works on arrays of any shape that
    broadcast together, e.g. (partition, site) wave arrays with (site,) angles. '''
    
    phi = dp - 90
    thet = ((angles + 90) % 360) - 90
    angle_term = ((phi - thet + 180) % 360) - 180                       # wrap to [-180, 180)
    angle_term = np.where(np.abs(angle_term) <= 90, angle_term, np.nan) # NaN where offshore-directed
    angle_term = angle_term * (np.pi / 180)

    cos_term = np.cos(angle_term)
    sin_term = np.sin(angle_term)
    qs = -k * (tp ** 0.2) * (hs ** 2.4) * (cos_term ** 1.2) * sin_term 
    return qs, angle_term


def get_dists_np(lon, lat):

    ''' Numpy version of get_dists_xr: haversine distance (m) from point i-1 to point i+1. NaN at the ends. '''

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    phi1, phi2 = lat_r[:-2], lat_r[2:]
    delta_phi = phi2 - phi1
    delta_lambda = lon_r[2:] - lon_r[:-2]

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = np.full(lat_r.shape, np.nan)
    distance[1:-1] = EARTH_RADIUS * c
    return distance


def get_delta_qs_np(qs):

    ''' Numpy version of get_delta_qs: qs[i-1] - qs[i+1]. NaN at the ends. '''

    dqs = np.full(qs.shape, np.nan)
    dqs[1:-1] = qs[:-2] - qs[2:]
    return dqs


def smooth(values, sigma):

    ''' Gaussian smoothing along site, same settings as the notebooks (mode='nearest'). '''

    return gaussian_filter1d(values, sigma=sigma, mode='nearest') if sigma > 0 else np.array(values, dtype=float)


def compute_deta_dt(lat, lon, dp, hs, tp, k, d_eff, gamma, shore_sigma, s_x, dist_sigma=0.0):

    ''' Steps 1-3 of the model for one time step.

    lat, lon     : (site,) shoreline from the previous step
    dp, hs, tp   : (partition, site) wave data for this time step, already mapped to sites
    s_x          : (site,) sources/sinks in m/s
    dist_sigma   : smoothing (points) of the shoreline used for distances in the gradient

    Angles come from the shoreline smoothed with shore_sigma, but distances come from the
    (at most lightly smoothed) actual shoreline: smoothing pulls points together around curves,
    which would overstate the gradient there and stop sediment from being conserved.

    Returns deta_dt (m/s) and the smoothed total Qs (m^3/s, calc_qs sign convention). '''

    # 1. Qs from each partition using shoreline angles from the (smoothed) previous shoreline
    lat_s = smooth(lat, shore_sigma)
    lon_s = smooth(lon, shore_sigma)
    angles = get_angles_np(lon_s, lat_s)

    qs_parts, _ = calc_qs_np(dp, hs, tp, angles, k)
    # an offshore-directed or missing partition moves no sediment, so it contributes zero
    qs_total = np.where(np.isfinite(qs_parts), qs_parts, 0.0).sum(axis=0)

    # 2. smooth Qs and take the gradient (grad = dQ_x/dx, see sign convention at the top)
    qs_smooth = smooth(qs_total, gamma)
    lat_d = lat_s if dist_sigma == shore_sigma else smooth(lat, dist_sigma)
    lon_d = lon_s if dist_sigma == shore_sigma else smooth(lon, dist_sigma)
    grad = get_delta_qs_np(qs_smooth) / get_dists_np(lon_d, lat_d)
    grad = np.where(np.isfinite(grad), grad, 0.0)  # ends: no transport gradient

    # 3. deta/dt = -(1/D_eff) dQ_x/dx + s_x
    deta_dt = -grad / d_eff + s_x
    return deta_dt, qs_smooth


# ======================================================================================
# geometry: eta (meters along transect) <-> lat/lon
# ======================================================================================

class TransectGeometry:

    ''' Converts beach width eta to lat/lon, the vectorized equivalent of
    line.interpolate(eta) + to_crs('EPSG:4326') in beach_w_to_gps.ipynb.

    CoastSat transects are straight two-point lines, so a point eta meters along the
    transect is origin + eta * unit_direction (in UTM). Unlike shapely's interpolate, this
    extrapolates past the transect ends instead of clamping, so eta < 0 or eta > length
    still moves the shoreline consistently. '''

    def __init__(self, transects_gdf_metric, site_ids, utm_crs):

        missing = pd.Index(site_ids).difference(transects_gdf_metric.index)
        if len(missing):
            raise KeyError(f"{len(missing)} sites have no transect geometry, e.g. {list(missing[:5])}")

        geoms = np.asarray(transects_gdf_metric.loc[site_ids].geometry.values)
        n_coords = shapely.get_num_coordinates(geoms)
        if np.any(n_coords > 2):
            warnings.warn(f"{int(np.sum(n_coords > 2))} transects have more than 2 vertices; "
                          "using the straight line from first to last vertex")

        start = shapely.get_point(geoms, 0)
        end = shapely.get_point(geoms, -1)
        self.x0, self.y0 = shapely.get_x(start), shapely.get_y(start)
        dx, dy = shapely.get_x(end) - self.x0, shapely.get_y(end) - self.y0
        self.length = np.hypot(dx, dy)
        self.ux, self.uy = dx / self.length, dy / self.length

        self.to_wgs84 = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)

    def eta_to_latlon(self, eta):
        x = self.x0 + eta * self.ux
        y = self.y0 + eta * self.uy
        lon, lat = self.to_wgs84.transform(x, y)
        return np.asarray(lat), np.asarray(lon)

    def origin_alongshore_dist(self):
        ''' Cumulative distance (m) between transect origins, used to interpolate missing initial eta. '''
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x0), np.diff(self.y0)))])


# ======================================================================================
# inputs
# ======================================================================================

def read_coastsat_csv(csv_path):

    ''' Reads one CoastSat shoreline csv the same way as beach_w_to_gps.ipynb. '''

    df = pd.read_csv(csv_path, header=None, usecols=[0, 1], names=['time', 'beach_width'])
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.tz_localize(None)
    df['beach_width'] = pd.to_numeric(df['beach_width'], errors='coerce')
    return df.dropna(subset=['time', 'beach_width'])


def load_initial_eta(site_ids, csv_dir, t0, method='median', window_days=365.0):

    ''' Initial beach width at time t0 for each site from the CoastSat csv files.

    Returns (eta, has_data): eta is NaN where no observation falls inside the window;
    has_data is False where the csv is missing or empty (those sites are dropped, matching
    the site list built in beach_w_to_gps.ipynb). '''

    t0 = pd.Timestamp(t0)
    half = pd.Timedelta(days=window_days / 2)
    eta = np.full(len(site_ids), np.nan)
    has_data = np.zeros(len(site_ids), dtype=bool)

    for j, site_id in enumerate(site_ids):
        csv_path = Path(csv_dir) / f"{site_id}.csv"
        if not csv_path.exists():
            continue
        df = read_coastsat_csv(csv_path)
        if df.empty:
            continue
        has_data[j] = True

        if method == 'nearest':
            gap = (df['time'] - t0).abs()
            if gap.min() <= half:
                eta[j] = df['beach_width'].iloc[gap.argmin()]
        elif method == 'median':
            in_window = df[(df['time'] >= t0 - half) & (df['time'] <= t0 + half)]
            if len(in_window):
                eta[j] = in_window['beach_width'].median()
        else:
            raise ValueError(f"eta_init_method must be 'median' or 'nearest', got {method!r}")

    return eta, has_data


def load_s_x(cfg, site_ids):

    ''' Sources/sinks per site in m/s. Sites missing from the file get 0. '''

    if cfg.s_x_file is None:
        return np.zeros(len(site_ids))

    obj = xr.open_dataset(cfg.path(cfg.s_x_file))
    da = obj[cfg.s_x_var] if isinstance(obj, xr.Dataset) else obj
    s_x = da.reindex(site=list(site_ids)).values.astype(float)
    n_missing = int(np.sum(~np.isfinite(s_x)))
    if n_missing:
        print(f"  s_x: {n_missing} of {len(site_ids)} sites missing from {cfg.s_x_file}, set to 0")
    s_x = np.where(np.isfinite(s_x), s_x, 0.0)
    obj.close()

    if cfg.s_x_units == 'm/yr':
        return s_x / SECONDS_PER_YEAR
    if cfg.s_x_units == 'm/s':
        return s_x
    raise ValueError(f"s_x_units must be 'm/yr' or 'm/s', got {cfg.s_x_units!r}")


class WaveForcing:

    ''' Lazily opened partitioned wave files, read in blocks of time and mapped to model sites. '''

    VARS = ('dp', 'hs', 'tp')

    def __init__(self, cfg):
        self.partitions = tuple(cfg.partitions)
        self.files = {}
        for var in self.VARS:
            for p in self.partitions:
                path = cfg.path(cfg.wave_files[var].format(i=p))
                ds = xr.open_dataset(path)
                if var not in ds.data_vars:
                    raise KeyError(f"{path} has no variable {var!r}; found {list(ds.data_vars)}")
                self.files[(var, p)] = ds

        ref = self.files[('dp', self.partitions[0])]
        self.times = ref['time'].values
        for (var, p), ds in self.files.items():
            if not np.array_equal(ds['time'].values, self.times):
                raise ValueError(f"time axis of {var}{p} differs from dp{self.partitions[0]}")

        self.wave_lat = ref['lat'].values
        self.wave_lon = ref['lon'].values
        self.mapping_idx = None

    def map_to_sites(self, shore_lat, shore_lon):
        ''' Nearest wave point for each shore point (fixed for the whole run). '''
        self.mapping_idx = get_nearest_indices(self.wave_lat, self.wave_lon, shore_lat, shore_lon)

    def read_block(self, start, stop):
        ''' Returns {var: array (partition, time, site)} for wave steps start..stop-1. '''
        out = {}
        for var in self.VARS:
            arr = np.empty((len(self.partitions), stop - start, len(self.mapping_idx)))
            for j, p in enumerate(self.partitions):
                vals = (self.files[(var, p)][var]
                        .isel(time=slice(start, stop))
                        .transpose('time', 'site')
                        .values)
                arr[j] = vals[:, self.mapping_idx]
            out[var] = arr
        return out

    def close(self):
        for ds in self.files.values():
            ds.close()


# ======================================================================================
# output
# ======================================================================================

class OutputWriter:

    ''' Appends model states to a netcdf file along an unlimited time dimension.

    The file opens with xr.open_dataset and has the same site/lat/lon structure as the
    other NC files: lat/lon are the initial shore points (site), eta is (time, site), and
    shore_lat/shore_lon hold the evolving shoreline. '''

    TIME_UNITS = 'hours since 1900-01-01 00:00:00'
    CALENDAR = 'standard'

    def __init__(self, path, site_ids=None, lat0=None, lon0=None, attrs=None,
                 save_latlon=True, save_diagnostics=True, mode='w'):

        self.path = Path(path)
        if mode == 'a':
            self.ds = nc.Dataset(self.path, 'a')
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            os.remove(self.path)

        n = len(site_ids)
        ds = nc.Dataset(self.path, 'w', format='NETCDF4')
        ds.createDimension('time', None)
        ds.createDimension('site', n)

        t = ds.createVariable('time', 'f8', ('time',))
        t.units, t.calendar = self.TIME_UNITS, self.CALENDAR

        site = ds.createVariable('site', str, ('site',))
        site[:] = np.array(site_ids, dtype=object)

        for name, vals, desc in [('lat', lat0, 'initial shoreline latitude'),
                                 ('lon', lon0, 'initial shoreline longitude')]:
            v = ds.createVariable(name, 'f8', ('site',))
            v[:] = vals
            v.long_name = desc

        chunks = (64, n)
        def add(name, units, desc):
            v = ds.createVariable(name, 'f8', ('time', 'site'), zlib=True, complevel=4,
                                  chunksizes=chunks, fill_value=np.nan)
            v.units, v.long_name, v.coordinates = units, desc, 'lat lon'

        add('eta', 'm', 'beach width (distance along transect from origin)')
        if save_latlon:
            add('shore_lat', 'degrees_north', 'shoreline latitude')
            add('shore_lon', 'degrees_east', 'shoreline longitude')
        if save_diagnostics:
            add('qs_mean', 'm3/s', 'smoothed total Qs averaged over the output interval (calc_qs sign convention)')
            add('deta_dt_mean', 'm/yr', 'deta/dt averaged over the output interval')

        for key, val in (attrs or {}).items():
            ds.setncattr(key, val if isinstance(val, (int, float, np.number)) else str(val))
        self.ds = ds

    def write(self, t, eta, lat=None, lon=None, qs_mean=None, deta_dt_mean=None):
        i = len(self.ds.dimensions['time'])
        self.ds['time'][i] = nc.date2num(pd.Timestamp(t).to_pydatetime(), self.TIME_UNITS, self.CALENDAR)
        self.ds['eta'][i, :] = eta
        if 'shore_lat' in self.ds.variables and lat is not None:
            self.ds['shore_lat'][i, :] = lat
            self.ds['shore_lon'][i, :] = lon
        if 'qs_mean' in self.ds.variables and qs_mean is not None:
            self.ds['qs_mean'][i, :] = qs_mean
            self.ds['deta_dt_mean'][i, :] = deta_dt_mean
        self.ds.sync()  # keep everything written so far safe if the job is killed

    def close(self):
        self.ds.close()

    @classmethod
    def read_last_state(cls, path):
        ''' Returns (time, eta, site_ids, lat0, lon0, attrs) of the last record in an output file. '''
        with nc.Dataset(path) as ds:
            ds.set_auto_mask(False)
            n = len(ds.dimensions['time'])
            if n == 0:
                raise ValueError(f"{path} has no records to restart from")
            t = nc.num2date(ds['time'][n - 1], cls.TIME_UNITS, cls.CALENDAR,
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            eta = np.array(ds['eta'][n - 1, :], dtype=float)
            site_ids = [str(s) for s in ds['site'][:]]
            lat0 = np.array(ds['lat'][:], dtype=float)
            lon0 = np.array(ds['lon'][:], dtype=float)
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        return np.datetime64(t, 'ns'), eta, site_ids, lat0, lon0, attrs


# ======================================================================================
# setup + time loop
# ======================================================================================

def select_sites(cfg, ids, has_data):

    ''' Keeps sites with CoastSat data (as in beach_w_to_gps), then crops to the configured domain. '''

    ids = np.asarray(ids)[has_data]

    def position(site):
        matches = np.where(ids == site)[0]
        if not len(matches):
            raise ValueError(f"domain site {site!r} not found among sites with CoastSat data")
        return int(matches[0])

    start = 0 if cfg.domain_start_site is None else position(cfg.domain_start_site) + cfg.domain_start_offset
    stop = len(ids) if cfg.domain_end_site is None else position(cfg.domain_end_site) + 1
    return list(ids[start:stop])


def run_model(cfg):

    ''' Runs the shoreline model from start_time to end_time and writes results to cfg.output_file. '''

    wall_start = time.time()
    k = cfg.k if cfg.k is not None else calc_alongshore_transport_k()

    print("Opening wave data...", flush=True)
    waves = WaveForcing(cfg)
    times = waves.times
    dts = np.diff(times).astype('timedelta64[ns]').astype(float) / 1e9
    dts = np.append(dts, np.median(dts))  # last record uses the typical spacing
    if not np.allclose(dts, dts[0]):
        print(f"  note: wave time spacing is not uniform ({dts.min():.0f}-{dts.max():.0f} s); "
              "each step uses its own spacing", flush=True)

    i_end = len(times) - 1 if cfg.end_time is None else \
        int(np.searchsorted(times, np.datetime64(cfg.end_time), side='right')) - 1

    print("Loading transects...", flush=True)
    transects_gdf_metric = gpd.read_file(cfg.path(cfg.transects_file)).to_crs(cfg.utm_crs).set_index('id')

    if cfg.restart and Path(cfg.output_file).exists():
        # --- resume from the last saved state
        t_last, eta, site_ids, lat0, lon0, attrs = OutputWriter.read_last_state(cfg.output_file)
        active = np.isfinite(eta)         # sites omitted earlier in the run are NaN in the saved state
        eta = np.where(active, eta, 0.0)  # placeholder only; omitted sites stay out of the calculation
        current = {'d_eff': cfg.d_eff, 'gamma': cfg.gamma, 'shore_sigma': cfg.shore_sigma,
                   'dist_sigma': cfg.dist_sigma, 'k': k, 'substeps': cfg.substeps}
        for key, val in current.items():
            if key in attrs and not np.isclose(float(attrs[key]), float(val)):
                warnings.warn(f"restart: {key} is {val} now but was {attrs[key]} in the saved run")
        matches = np.where(times == t_last)[0]
        if not len(matches):
            raise ValueError(f"restart time {t_last} is not on the wave time axis")
        i_start = int(matches[0])
        geom = TransectGeometry(transects_gdf_metric, site_ids, cfg.utm_crs)
        writer = OutputWriter(cfg.output_file, mode='a')
        print(f"Restarting from {pd.Timestamp(t_last)} (wave step {i_start})", flush=True)

    else:
        # --- fresh start
        i_start = 0 if cfg.start_time is None else \
            int(np.searchsorted(times, np.datetime64(cfg.start_time), side='left'))
        t0 = times[i_start]

        print(f"Loading initial eta from CoastSat ({cfg.eta_init_method}, "
              f"{cfg.eta_init_window_days:g}-day window around {pd.Timestamp(t0).date()})...", flush=True)
        all_ids = list(transects_gdf_metric.index)
        eta_all, has_data = load_initial_eta(all_ids, cfg.path(cfg.eta_init_dir), t0,
                                             cfg.eta_init_method, cfg.eta_init_window_days)
        site_ids = select_sites(cfg, all_ids, has_data)
        eta = pd.Series(eta_all, index=all_ids).loc[site_ids].values  # same order as site_ids

        geom = TransectGeometry(transects_gdf_metric, site_ids, cfg.utm_crs)
        n_fill = int(np.sum(~np.isfinite(eta)))
        if n_fill == len(eta):
            raise ValueError("no CoastSat observations in the initial-eta window; widen eta_init_window_days")
        if n_fill:
            s = geom.origin_alongshore_dist()
            ok = np.isfinite(eta)
            eta = np.where(ok, eta, np.interp(s, s[ok], eta[ok]))
            print(f"  {n_fill} of {len(eta)} sites had no observation in the window; "
                  "interpolated from neighbors along shore", flush=True)

        active = np.ones(len(eta), dtype=bool)  # sites still in the calculation
        lat0, lon0 = geom.eta_to_latlon(eta)
        attrs = {'d_eff': cfg.d_eff, 'gamma': cfg.gamma, 'shore_sigma': cfg.shore_sigma,
                 'dist_sigma': cfg.dist_sigma, 'k': k,
                 'substeps': cfg.substeps, 's_x_file': cfg.s_x_file, 's_x_units': cfg.s_x_units,
                 'partitions': str(cfg.partitions), 'eta_init_method': cfg.eta_init_method,
                 'eta_init_window_days': cfg.eta_init_window_days,
                 'start_time': str(pd.Timestamp(t0)), 'created': str(pd.Timestamp.now())}
        writer = OutputWriter(cfg.output_file, site_ids, lat0, lon0, attrs,
                              cfg.save_latlon, cfg.save_diagnostics, mode='w')
        writer.write(t0, eta, lat0, lon0)

    # wave points are mapped once, to the smoothed initial shoreline (as in calculate_qs_partitions)
    waves.map_to_sites(smooth(lat0, cfg.shore_sigma), smooth(lon0, cfg.shore_sigma))
    s_x = load_s_x(cfg, site_ids)
    lat, lon = geom.eta_to_latlon(eta)

    n_steps = i_end - i_start + 1
    spacing = np.nanmean(get_dists_np(lon0, lat0)) / 2
    print(f"\n{len(site_ids)} sites | mean spacing {spacing:.1f} m | "
          f"gamma {cfg.gamma:g} pts (~{cfg.gamma * spacing:.0f} m) | "
          f"shore_sigma {cfg.shore_sigma:g} pts (~{cfg.shore_sigma * spacing:.0f} m) | "
          f"dist_sigma {cfg.dist_sigma:g} pts")
    print(f"D_eff {cfg.d_eff:g} m | k {k:.4g} | s_x range {s_x.min() * SECONDS_PER_YEAR:.3g} "
          f"to {s_x.max() * SECONDS_PER_YEAR:.3g} m/yr")
    print(f"{n_steps} wave steps from {pd.Timestamp(times[i_start])} to {pd.Timestamp(times[i_end])}, "
          f"{cfg.substeps} substep(s) each\n", flush=True)

    eta_ref = eta.copy()
    qs_acc = np.zeros_like(eta)
    deta_acc = np.zeros_like(eta)
    n_acc = 0
    n_sites = len(site_ids)
    omitted_log = []        # (site, time, change in m) for every site dropped from the run
    loop_start = time.time()
    if not active.all():
        print(f"  {int((~active).sum())} site(s) were omitted earlier in this run and stay omitted\n", flush=True)

    try:
        for b0 in range(i_start, i_end + 1, cfg.wave_block):
            b1 = min(b0 + cfg.wave_block, i_end + 1)
            block = waves.read_block(b0, b1)

            for j in range(b1 - b0):
                n = b0 + j
                dt = dts[n] / cfg.substeps
                eta_before, lat_before, lon_before = eta, lat, lon

                while True:
                    eta, lat, lon = eta_before, lat_before, lon_before
                    qs_step = np.zeros(n_sites)
                    deta_step = np.zeros(n_sites)

                    # omitted sites are left out of the calculation entirely, so the sites on either
                    # side of them become neighbors for the angles, distances, smoothing and gradient
                    take = slice(None) if active.all() else active

                    for _ in range(cfg.substeps):
                        # steps 1-3
                        deta_act, qs_act = compute_deta_dt(
                            lat[take], lon[take], block['dp'][:, j][:, take], block['hs'][:, j][:, take],
                            block['tp'][:, j][:, take], k, cfg.d_eff, cfg.gamma, cfg.shore_sigma,
                            s_x[take], cfg.dist_sigma)
                        deta_dt = np.zeros(n_sites)
                        deta_dt[take] = deta_act
                        qs_smooth = np.zeros(n_sites)
                        qs_smooth[take] = qs_act
                        # step 4
                        eta = eta + deta_dt * dt
                        # step 5
                        lat, lon = geom.eta_to_latlon(eta)

                        qs_step += qs_smooth
                        deta_step += deta_dt

                    # a site that jumps too far (or goes non-finite) is omitted for the rest of the run
                    change = np.abs(eta - eta_before)
                    bad = active & (~np.isfinite(eta) | (change > cfg.max_step_change))
                    if not bad.any():
                        break

                    for i in np.flatnonzero(bad):
                        omitted_log.append((site_ids[i], pd.Timestamp(times[n]), float(change[i])))
                        if len(omitted_log) <= 20:
                            what = ("became non-finite" if not np.isfinite(change[i])
                                    else f"changed by {change[i]:.2f} m")
                            print(f"  omitting site {site_ids[i]} at {pd.Timestamp(times[n])}: "
                                  f"eta {what} in one step", flush=True)
                        elif len(omitted_log) == 21:
                            print("  (further omissions are not printed; they are listed at the end)", flush=True)
                    active = active & ~bad

                    n_omitted = n_sites - int(active.sum())
                    if n_omitted / n_sites > cfg.max_omitted_fraction or int(active.sum()) < 3:
                        raise RuntimeError(
                            f"{n_omitted} of {n_sites} sites ({n_omitted / n_sites:.1%}) have been omitted by "
                            f"{pd.Timestamp(times[n])}, more than max_omitted_fraction "
                            f"({cfg.max_omitted_fraction:.0%}). The run is probably numerically unstable: try "
                            f"more substeps, larger gamma/shore_sigma, or check the inputs at this time. "
                            f"First omitted: {[o[0] for o in omitted_log[:5]]}")
                    # go around again: redo this step without the omitted sites so that their
                    # blown-up values don't leak into the neighbors through the smoothing

                qs_acc += qs_step
                deta_acc += deta_step
                n_acc += cfg.substeps

                t_state = times[n] + np.timedelta64(int(dts[n] * 1e9), 'ns')
                done = n - i_start + 1
                if done % cfg.output_every == 0 or n == i_end:
                    # omitted sites are written as NaN from the time they were dropped
                    writer.write(t_state,
                                 np.where(active, eta, np.nan),
                                 np.where(active, lat, np.nan), np.where(active, lon, np.nan),
                                 np.where(active, qs_acc / n_acc, np.nan),
                                 np.where(active, deta_acc / n_acc * SECONDS_PER_YEAR, np.nan))
                    qs_acc[:] = 0
                    deta_acc[:] = 0
                    n_acc = 0

                if done % cfg.progress_every == 0 or n == i_end:
                    elapsed = time.time() - loop_start
                    remaining = elapsed / done * (n_steps - done)
                    print(f"Step {done}/{n_steps} | {pd.Timestamp(t_state)} | "
                          f"max |eta change| {np.max(np.abs(eta - eta_ref)[active]):.2f} m | "
                          f"elapsed {elapsed / 60:.1f} min | ~{remaining / 60:.1f} min left", flush=True)
    finally:
        writer.close()
        waves.close()

    if omitted_log:
        print(f"\n{len(omitted_log)} site(s) omitted during this run (NaN in the output from that time on):")
        for site, t_omit, change in omitted_log[:50]:
            print(f"  {site}  {t_omit}  (changed by {change:.2f} m)")
        if len(omitted_log) > 50:
            print(f"  ... and {len(omitted_log) - 50} more")

    print(f"\nDone! :) Saved {cfg.output_file} ({(time.time() - wall_start) / 60:.1f} min total)")
    return cfg.output_file
