#!/bin/bash

model=$1
mem_profile_dir=$2
comp_profile_dir=$3
gpu_reuse_pause_time=${4:-"300"}

if [ $# -lt 3 ]; then
  echo "[USAGE] [model] [memory profile dir] [comp profile dir] [GPU reuse pause time (default: 300 sec)]"
  exit 1
fi

python profiler/comp/comp_profiler_driver_with_max_mem.py \
    -a $model \
    --mem-profile-dir $mem_profile_dir \
    --profile-dir $comp_profile_dir \
    -p $gpu_reuse_pause_time