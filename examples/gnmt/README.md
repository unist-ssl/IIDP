# GNMT
The original code of this example comes from [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/b09ce9831bfa296dd56eaa3e2516eb2b76db010e/PyTorch/Translation/GNMT).

## Table of Contents

<!-- TOC GFM -->

* [Prerequisites](#prerequisites)
* [How to run on IIDP](#how-to-run-on-iidp)
* [Configuration of IIDP](#configuration-of-iidp)
  * [1. Memory profiler](#1-memory-profiler)
  * [2. Computation profiler](#2-computation-profiler)
  * [3. DDP bucket profiler](#3-ddp-bucket-profiler)
  * [4. Synchronize profile data](#4-synchronize-computation-and-memory-profile-data-for-heterogeneous-gpus)
  * [5. Configuration solver](#5-configuration-solver)

<!-- /TOC -->

## Prerequisites
- Software packages
  ```bash
  pip install -r requirements.txt
  ```
- WMT16 en-de dataset

  The detailed procedure in ```prepare_dataset.sh``` is explained in [this link](https://github.com/NVIDIA/DeepLearningExamples/tree/b09ce9831bfa296dd56eaa3e2516eb2b76db010e/PyTorch/Translation/GNMT#getting-the-data).

  Dataset is stored to ```IIDP_DATA_STORAGE/pytorch_wmt16_en_de```.
  ```
  ./scripts/utils/wmt16_en_de/prepare_dataset.sh
  ```

## How to run on IIDP
- Benchmark (throughput)
  ```
  ./scripts/distributed/distributed_run.sh [node rank] [number of nodes] [local batch size] [number of VSWs] [accum step (default: 0)] [weight sync method (default: recommend)] [master (optional)]
  ```
- Convergence
  ```
  ./scripts/convergence/distributed/distributed_run.sh [node rank] [number of nodes] [local batch size] [number of VSWs] [accum step (default: 0)] [weight sync method (default: recommend)] [master (optional)] [log file (optional)]
  ```
- Example
  - node0
    ```
    ./scripts/distributed/distributed_run.sh 0 2 32 3
    ```
  - node1
    ```
    ./scripts/distributed/distributed_run.sh 1 2 32 1
    ```


## Configuration of IIDP
We provide the configuration solver that finds the best configuration of IIDP with repect to 1) local batch size 2) the number of VSWs 3) gradient accumulation steps.

To get the configuration, the following steps are required.

### 1. Memory Profiler
Memory profiler finds 1) the ranges of local batch size 2) the maximum number of VSWs with each local batch size.
If LBS search mode is 'static', the user-specified local batch size is only searched.

Generate memory profile data on each node.
```
./profiler/memory/scripts/run_mem_profiler.sh [Local Batch Size] [profile dir (default: mem_profile_data)] [LBS search mode (default: dynamic)]
```
Example
```
# On node0, node1
./profiler/memory/scripts/run_mem_profiler.sh 32
```

### 2. Computation Profiler
Generate computation profile data on each node with memory profile data.

For each server, execute computation profiler to get profile data json file.

GPU pause time is set to 300 seconds by default because the temperature of the GPUs can affect the accuracy of time measurement, potentially impacting the performance model.
  ```
  ./profiler/comp/scripts/run_comp_profiler_with_max_mem.sh [local mem profile dir] [local comp profile dir] [GPU reuse pause time (default: 300 sec)]
  ```
Example
```
# On node0, node1
./profiler/comp/scripts/run_comp_profiler_with_max_mem.sh mem_profile_data comp_profile_data
```

### 3. DDP Bucket Profiler
Generate profile data on each node.
```
./profiler/ddp_bucket/scripts/run_bucket_profiler.sh [profile dir (default: bucket_profile_data)] [plot dir (default: bucket_profile_plot)] [visible GPU ID (optional)]
```
Example
```
# On node0, node1
./profiler/ddp_bucket/scripts/run_bucket_profiler.sh
```

### 4. Synchronize computation and memory profile data for heterogeneous GPUs
All-gather and broadcast profile data results on every node.

After finishing the profile step on every node, execute the below script.
  ```
  ./profiler/utils/sync/sync_data_across_servers.sh [local profile dir] [all-gathered profile dir] [master node] [slave nodes]
  ```
Example
```
# On node0, node1
./profiler/utils/sync/sync_data_across_servers.sh mem_profile_data cluster_mem_profile_data node0 node1
./profiler/utils/sync/sync_data_across_servers.sh comp_profile_data cluster_comp_profile_data node0 node1
```

### 5. Configuration Solver
This step is unnecessary on every node, just one node is enough.

5.1. Prepare config file (JSON)
- Format of configuration file
  ```json
  {
    "memory_profile_dir": "{mem profile dir}",
    "comp_profile_dir": "{comp profile dir}",
    "comm_profile_dir": "{comm profile dir}",
    "bucket_profile_dir": "{bucket profile dir}",
    "gpu_cluster_info": "{gpu cluster info file (JSON)}",
    "available_servers": ["{first node}", "{second node}"]
  }
  ```
  Example config file: ```iidp_config/example.json```
  ```json
  {
    "memory_profile_dir": "cluster_mem_profile_data",
    "comp_profile_dir": "cluster_comp_profile_data",
    "comm_profile_dir": "../common/comm_profile_data",
    "bucket_profile_dir": "bucket_profile_data",
    "gpu_cluster_info": "../common/cluster_config/example_cluster_info.json",
    "available_servers": ["node0", "node1"]
  }
  ```

5.2. Run
  ```bash
  ./scripts/config/run_config_solver.sh [config file (JSON)] [global batch size] [weight sync method (default: recommend)]
  ```
  Example
  ```bash
  ./scripts/config/run_config_solver.sh iidp_config/example.json 512
  ```