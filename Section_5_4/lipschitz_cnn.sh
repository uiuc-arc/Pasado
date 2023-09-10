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


if [ $l_flag -eq 1 ]; then
  python3 get_lipschitz_cnn.py --net small -l
  python3 get_lipschitz_cnn.py --net med -l
  python3 get_lipschitz_cnn.py --net big -l
else
  python3 get_lipschitz_cnn.py --net small
  python3 get_lipschitz_cnn.py --net med
fi
