#!/bin/bash

model=$1
profile_dir=${2:-"bucket_profile_data"}
visible_gpu=${3:-"0"}

if [ $# -lt 1 ]; then
  echo "[USAGE] [model] [profile dir (default: bucket_profile_data)] [visible GPU ID (optional)]"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$visible_gpu python profiler/ddp_bucket/main.py \
  -a $model \
  --profile-dir $profile_dir
