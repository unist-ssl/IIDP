#!/bin/bash

rank=$1
world_size=$2
local_batch_size=$3
num_models=$4
accum_step=${5:-0}
weight_sync_method=${6:-"recommend"}
master=$7
log_file=$8

if [ $# -lt 4 ]; then
  echo "[USAGE] [rank] [number of nodes] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)] [log file (optional)]"
  exit 1
fi
echo "local batch size: "$local_batch_size
echo "number of VSWs: "$num_models
echo "number of GA steps: "$accum_step

if [ $world_size == "1" ]; then
  echo "Single-node training"
  master=`hostname`
fi
if [ -z $master ]; then
  master=$IIDP_MAIN_NODE
fi

dataset_dir=$IIDP_DATA_STORAGE'/coco2017/'

model='fasterrcnn_resnet50_fpn'

# To avoid the error: "Too many open files"
ulimit -n 65535

CMD="
  python train.py \
    --dist-url 'tcp://'${master}':22222' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size $world_size \
    --rank $rank \
    --model $model \
    -lbs $local_batch_size \
    --num-models $num_models \
    --accum-step $accum_step \
    --weight-sync-method $weight_sync_method \
    $dataset_dir
"

if [ $rank == "0" ]; then
  if [ -z $log_file ]; then
    timestamp=`date +%Y%m%d_%H%M%S`
    dir_name=${timestamp}_convergence_log
    mkdir $dir_name
    log_file=$dir_name/"convergence_log.txt"
  fi
  echo "log file: "$log_file
  eval $CMD 2>&1 | tee -a -i $log_file
else
  eval $CMD
fi
