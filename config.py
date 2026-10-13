'''
config.py

Model settings. Edit this file, then run:  python run_model.py
See ModelConfig in shoreline_model.py for every option and its default.
'''

from pathlib import Path
from shoreline_model import ModelConfig

BASE_DIR = Path('/work/acd99/shoreline_evolution')

CONFIG = ModelConfig(

    # --- inputs
    data_dir=BASE_DIR / 'data',
    wave_files={
        'dp': 'dp{i}_CenturyHindcast_500m.nc',
        'hs': 'phs{i}_CenturyHindcast_500m.nc',
        'tp': 'ptp{i}_CenturyHindcast_500m.nc',
    },
    s_x_file=None,              # change to actual s_x when done
    s_x_var='s_x',              # name of the variable inside s_x_file
    s_x_units='m/yr',

    # --- output
    output_file=BASE_DIR / 'output' / 'shoreline_model_output.nc',
    output_every=30,            # daily for daily wave data change to 24 for hourly wave data

    # --- time (None = full wave record)
    start_time='1984-01-01',            # e.g. '1984-01-01' (datetime)
    end_time=None,
    substeps=1,

    # --- domain
    domain_start_site=None,
    domain_start_offset=0,
    domain_end_site=None,

    # --- physics
    d_eff=10.0,                 # effective shoreface depth (m)
    gamma=25.0,                 # Qs smoothing width (points)
    shore_sigma=0.0,            # shoreline smoothing width before computing angles (points)
    k=0.01,                     # transport coefficient

    # --- initial eta from CoastSat
    eta_init_method='median',
    eta_init_window_days=365.0,
)
