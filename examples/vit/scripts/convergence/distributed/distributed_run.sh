#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
local_batch_size=$3
num_models=$4
accum_step=${5:-"0"}
weight_sync_method=${6:-"recommend"}
master=$7
log_file=$8

if [ $# -lt 4 ]; then
  echo "[USAGE] [rank] [number of nodes] [local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)] [log file (optional)]"
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

lr=0.0005 #128 - 1e-3, 64 - 5e-4
epochs=90
timestamp=`date +%Y%m%d_%H%M%S`
num_gpus_in_server=`nvidia-smi --list-gpus | wc -l`
CMD="
  python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus_in_server \
    --nnodes=$world_size \
    --node_rank=$rank \
    --master_addr=$master \
    --master_port=29000 \
    main.py \
      --model t2t_vit_14 \
      -lbs $local_batch_size \
      --num-models $num_models \
      --accum-step $accum_step \
      --weight-sync-method $weight_sync_method \
      --lr $lr \
      --weight-decay .05 \
      --img-size 224 \
      --epochs $epochs \
      $data_dir
"
if [ $rank == "0" ]; then
  if [ -z $log_file ]; then
    dir_name=${timestamp}_vit_convergence_log
    mkdir $dir_name
    log_file=$dir_name/"convergence_log.txt"
  fi
  echo "log file: "$log_file
  eval $CMD 2>&1 | tee -a -i $log_file
else
  eval $CMD
fi
