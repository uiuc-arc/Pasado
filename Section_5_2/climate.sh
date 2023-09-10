#!/bin/sh

mkdir -p img data
python3 climate_ode_script.py
python3 climate_plotter.py
python3 climate_ratio_scatter.py
