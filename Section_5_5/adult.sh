#!/bin/sh

l_flag=0

while getopts "l" opt; do
  case $opt in
    l)
      l_flag=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

mkdir -p img data

if [ $l_flag -eq 1 ]; then
  python3 adult_script.py -l
else
  python3 adult_script.py
fi

python3 adult_plot.py
