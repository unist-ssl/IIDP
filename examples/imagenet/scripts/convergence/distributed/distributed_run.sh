#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
model=$3
local_batch_size=$4
num_models=$5
accum_step=${6:-"0"}
weight_sync_method=${7:-"recommend"}
master=$8
log_file=$9

if [ $# -lt 5 ]; then
  echo "[USAGE] [rank] [number of nodes] [model] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)] [log file (optional)]"
  exit 1
fi

if [ $world_size == "1" ]; then
  echo "Single-node training"
  master=`hostname`
fi
if [ -z $master ]; then
  master=$IIDP_MAIN_NODE
fi

data_dir=$IIDP_DATA_STORAGE/imagenet
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

epochs=90
timestamp=`date +%Y%m%d_%H%M%S`
CMD="
  python main.py -a $model \
    --dist-url 'tcp://'${master}':32412' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size $world_size \
    --rank $rank \
    -lbs $local_batch_size \
    --num-models $num_models \
    --accum-step $accum_step \
    --epochs $epochs \
    --lr-scaling \
    --weight-sync-method $weight_sync_method \
    $data_dir
"
if [ $rank == "0" ]; then
  if [ -z $log_file ]; then
    dir_name=${timestamp}_${model}_imagenet_convergence_log
    mkdir $dir_name
    log_file=$dir_name/"convergence_log.txt"
  fi
  echo "log file: "$log_file
  eval $CMD 2>&1 | tee -a -i $log_file
else
  eval $CMD
fi
