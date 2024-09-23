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

data_dir=$IIDP_DATA_STORAGE/imagenet
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

num_minibatches=100
lr=0.00025 #128 - 1e-3, 64 - 5e-4, 32 - 25e-5
num_gpus_in_server=`nvidia-smi --list-gpus | wc -l`
python -m torch.distributed.launch \
  --nproc_per_node=$num_gpus_in_server \
  --nnodes=$world_size \
  --node_rank=$rank \
  --master_addr=$master \
  --master_port=29001 \
  main.py \
    --model t2t_vit_14 \
    -lbs $local_batch_size \
    --num-models $num_models \
    --accum-step $accum_step \
    --weight-sync-method $weight_sync_method \
    --lr $lr \
    --weight-decay .05 \
    --img-size 224 \
    --num-minibatches $num_minibatches \
    --no-validate \
    $data_dir

