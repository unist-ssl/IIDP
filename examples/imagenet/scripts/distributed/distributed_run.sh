#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
model=$3
local_batch_size=$4
num_models=$5
accum_step=${6:-"0"}
weight_sync_method=${7:-"recommend"}
master=$8

if [ $# -lt 5 ]; then
  echo "[USAGE] [rank] [number of nodes] [model] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)]"
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

num_minibatches=100

python main.py -a $model \
  --dist-url 'tcp://'${master}':32005' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size $world_size \
  --rank $rank \
  -lbs $local_batch_size \
  --num-models $num_models \
  --accum-step $accum_step \
  --num-minibatches $num_minibatches \
  --no-validate \
  --lr-scaling \
  --weight-sync-method $weight_sync_method \
  $data_dir

