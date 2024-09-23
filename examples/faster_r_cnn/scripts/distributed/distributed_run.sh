#!/bin/bash

rank=$1
world_size=$2
local_batch_size=$3
num_models=$4
accum_step=${5:-0}
weight_sync_method=${6:-"recommend"}
master=$7

if [ $# -lt 4 ]; then
  echo "[USAGE] [rank] [number of nodes] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)]"
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
num_minibatches=100

# To avoid the error: "Too many open files"
ulimit -n 65535

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
  --num-minibatches $num_minibatches \
  --no-validate \
  --weight-sync-method $weight_sync_method \
  $dataset_dir
