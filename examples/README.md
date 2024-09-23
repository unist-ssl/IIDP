# How to run example on IIDP

## Common setup
Take an example of the hardware setup of two GPU nodes.
```
Hostname: node0
GPU: 4 x Tesla V100-PCIE-16GB (TFPLOS: 14.13)
PCIe bandwidth: 15.75 GB/s
InfiniBand (IB) bandwidth: 100Gbps
```
```
Hostname: node1
GPU: 4 x Tesla P100-PCIE-16GB (TFPLOS: 9.52)
PCIe bandwidth: 15.75 GB/s
InfiniBand (IB) bandwidth: 100Gbps
```

**[Important]** The result of following procedures must exist on **every node**.

1. Set main node and dataset storage path on every node
    ```bash
    export IIDP_MAIN_NODE=<main node hostname>
    export IIDP_DATA_STORAGE=<dataset storage path>
    ```
    Example configuration
    ```
    export IIDP_MAIN_NODE=node0
    export IIDP_DATA_STORAGE=/mnt/dataset
    ```

[NOTE] Procedures 2 and 3 are required for the configuration of IIDP.

2. Prepare cluster config file on every node

    To get the type of GPU, execute the below command.
    ```
    python -c "import torch; print(torch.cuda.get_device_name())"
    ```
    To get the hostname, execute the below command.
    ```
    echo `hostname`
    ```
    In this case, your cluster config file (e.g, ```common/cluster_config/cluster_info.json```) should be configured as follows.

    The unit of ```intra_network_bandwidth``` and ```inter_network_bandwidth``` is 'Bytes per seconds'.
    ```json
    {
        "node0": {
            "type": "Tesla V100-PCIE-16GB",
            "tfplos": 14.13,
            "number": 4,
            "intra_network_bandwidth": 15750000000,
            "inter_network_bandwidth": 12500000000
        },
        "node1": {
            "type": "Tesla P100-PCIE-16GB",
            "tfplos": 9.52,
            "number": 4,
            "intra_network_bandwidth": 15750000000,
            "inter_network_bandwidth": 12500000000
        }
    }
    ```

3. Profile communication (all-reduce)

    How to run

    GPU pause time is set to 300 seconds by default because the temperature of the GPUs can affect the accuracy of time measurement, potentially impacting the performance model.
    ```
    ./profiler/comm/scripts/get_comm_profile_data.sh [rank] [number of servers] [master] [intra (bytes/sec)] [inter (bytes/sec)] [profile dir] [GPU pause time (default: 300 sec)]
    ```

    Intra-node
    ```
    # node0 (main node)
    cd $IIDP_HOME/examples/common
    ./profiler/comm/scripts/get_comm_profile_data.sh 0 1 node0 15750000000 12500000000 comm_profile_data
    ```

    Inter-node
    ```
    # node0 (main node)
    cd $IIDP_HOME/examples/common
    ./profiler/comm/scripts/get_comm_profile_data.sh 0 2 node0 15750000000 12500000000 comm_profile_data

    # node1
    cd $IIDP_HOME/examples/common
    ./profiler/comm/scripts/get_comm_profile_data.sh 1 2 node0 15750000000 12500000000 comm_profile_data
    ```

    Make sure that result on node0 exists at ```examples/common/comm_profile_data```
    ```
    comm_profile_data/
        ├── inter_comm_profile_data.txt
        └── intra_comm_profile_data.txt
    ```

   Send ```comm_profile_data/``` to ```node1``` so that result should be placed on node0 and node1
   ```
   scp -r comm_profile_data node1:$IIDP_HOME/examples/common
   ```

## PyTorch Examples
- [ImageNet](imagenet/)
- [ViT](vit/)
- [Faster-R-CNN](faster_r_cnn/)
- [GNMT](gnmt/)