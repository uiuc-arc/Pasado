#!/bin/sh

mkdir -p img data
l_flag=0
f_flag=0

while getopts "lf" opt; do
  case ${opt} in
  l)
    l_flag=1
    ;;
  f)
    f_flag=1
    ;;
  \?)
    echo "Invalid option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done

if [ $l_flag -eq 1 ] && [ $f_flag -eq 1 ]; then
  python3 chemical_ode_script.py -l -f
elif [ $l_flag -eq 1 ]; then
  python3 chemical_ode_script.py -l
elif [ $f_flag -eq 1 ]; then
  python3 chemical_ode_script.py -f
else
  python3 chemical_ode_script.py
fi

if [ $f_flag -eq 1 ]; then
  python3 chemical_plotter.py -f
else
  python3 chemical_plotter.py
fi

python3 chemical_ratio_scatter.py
