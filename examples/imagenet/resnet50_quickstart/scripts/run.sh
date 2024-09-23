#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
local_batch_size=$3
num_models=$4
accum_step=${5:-"0"}
weight_sync_method=${6:-"recommend"}
master=$7

if [ $# -lt 4 ]; then
  echo "[USAGE] [rank] [number of nodes] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)]"
  exit 1
fi

if [ $world_size == "1" ]; then
  echo "Single-node training"
  master=`hostname`
fi
if [ -z $master ]; then
  master=$IIDP_MAIN_NODE
fi

num_minibatches=100

python main.py -a 'resnet50' \
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
  --synthetic-dataset
