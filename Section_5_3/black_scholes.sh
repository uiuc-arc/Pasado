#!/bin/sh

mkdir -p img data
python3 black_scholes_rev_script.py
python3 black_scholes_ratio_scatter.py
