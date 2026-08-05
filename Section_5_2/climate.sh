#!/bin/sh

export PYTHONPATH="$(cd "$(dirname "$0")/../forward_mode_non_tensorized_src" && pwd):$PYTHONPATH"

mkdir -p img data
python3 climate_ode_script.py
python3 climate_plotter.py
python3 climate_ratio_scatter.py
