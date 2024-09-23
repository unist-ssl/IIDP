#!/bin/bash

rank=$1
world_size=$2  # Number of nodes
local_batch_size=$3
num_models=$4
accum_step=${5:-0}
weight_sync_method=${6:-"recommend"}
master=$7
log_file=$8

if [ $# -lt 4 ]; then
  echo "[USAGE] [rank] [number of nodes] [Local batch size] [number of VSWs] [accum step] [weight sync method (default: recommend)] [master (optional)] [log file (optional)]"
  exit 1
fi

if [ $world_size == "1" ]; then
  echo "Single-node training"
  master=`hostname`
fi
if [ -z $master ]; then
  master=$IIDP_MAIN_NODE
fi

data_dir=$IIDP_DATA_STORAGE/pytorch_wmt16_en_de
if [ ! -d $data_dir ]; then
  echo "No such data dir:"$data_dir
  exit 1
fi

num_layers=4
echo "Number of layers: "$num_layers
epochs=20

timestamp=`date +%Y%m%d_%H%M%S`
save_dir=$IIDP_DATA_STORAGE/../${timestamp}_gnmt_ckpt
eval_dir=${timestamp}_gnmt_eval_dir
CMD="
  python train.py \
    --dist-url 'tcp://'${master}':24543' \
    --dist-backend 'nccl' \
    --world-size $world_size \
    --rank $rank \
    --dataset-dir $data_dir \
    --math fp32 \
    --seed 2 \
    --num-layers $num_layers \
    --local-batch-size $local_batch_size \
    --num-models $num_models \
    --accum-step $accum_step \
    --weight-sync-method $weight_sync_method \
    --target-bleu 24.61 \
    --epochs $epochs \
    --save-dir $save_dir \
    --eval-dir $eval_dir
"

# To avoid the error: "Too many open files"
ulimit -n 65535

if [ $rank == "0" ]; then
  if [ -z $log_file ]; then
    dir_name=${timestamp}_gnmt_convergence_log
    mkdir $dir_name
    log_file=$dir_name/convergence_log.log
  fi
  echo "log file: "$log_file
  eval $CMD 2>&1 | tee -a -i $log_file
else
  eval $CMD
fi
