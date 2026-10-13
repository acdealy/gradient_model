'''
run_model.py

Runs the shoreline model with the settings in config.py.

    python run_model.py                  # fresh run
    python run_model.py --restart        # continue from the last state in the output file
    python run_model.py --end 1990-01-01 # override end_time without editing config.py

For long runs on the cluster, run it detached so it survives a dropped connection:
    nohup python run_model.py > model.log 2>&1 &
'''

import argparse
from dataclasses import replace

from config import CONFIG
from shoreline_model import run_model

parser = argparse.ArgumentParser(description='Run the NC shoreline evolution model.')
parser.add_argument('--restart', action='store_true', help='resume from the last saved state')
parser.add_argument('--start', default=None, help='override start_time, e.g. 1984-01-01')
parser.add_argument('--end', default=None, help='override end_time, e.g. 1990-01-01')
args = parser.parse_args()

cfg = CONFIG
if args.restart:
    cfg = replace(cfg, restart=True)
if args.start:
    cfg = replace(cfg, start_time=args.start)
if args.end:
    cfg = replace(cfg, end_time=args.end)

run_model(cfg)
