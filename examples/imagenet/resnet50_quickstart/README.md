# QuickStart for ResNet-50

We assume that current path is ```$IIDP_HOME/examples/imagenet```.

We provide 3 types of profile data on the below hardware setup:
1) Computation: ```resnet50_quickstart/cluster_comp_profile_data```
2) Memory: ```resnet50_quickstart/cluster_mem_profile_data```
3) Communication (All-reduce): ```resnet50_quickstart/comm_profile_data``` and ```resnet50_quickstart/bucket_profile_data```

## Hardware setup
Assume the hardware setup of two GPU nodes.
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

## Software setup
```bash
examples/imagenet$ pip install -r requirements.txt
```

## Run configuration solver for global batch size of 512
Execute the below command to reproduce the result.
```bash
examples/imagenet$ ./scripts/config/run_config_solver.sh resnet50_quickstart/config.json 512
...
================================================================================================================================
[INFO] Solution - GBS: 512 | LBS: 32 | weight sync method: overlap | config: ['node0:4GPU,VSW:3,GA:0', 'node1:4GPU,VSW:1,GA:0']
================================================================================================================================
```
Output of IIDP configuration indicates:
1) Local Batch Size (LBS) is ```32```.
2) One-way weight synchronization method is ```overlapping```.
3) For each GPU in  ```node0 (V100)```, the number of VSWs is ```3``` and the GA step is ```0```.
4) For each GPU in ```node1 (P100)```, the number of VSWs is ```1``` and the GA step is ```0```.

## Run ResNet-50 on IIDP
The script executes to train ResNet-50 for 100 iterations on IIDP with synthetic dataset.

On node0:
```bash
examples/imagenet$ export IIDP_MAIN_NODE=<User V100 server hostname>
examples/imagenet$ ./resnet50_quickstart/scripts/run.sh 0 2 32 3 0 overlap
```

On node1:
```bash
examples/imagenet$ export IIDP_MAIN_NODE=<User V100 server hostname>
examples/imagenet$ ./resnet50_quickstart/scripts/run.sh 1 2 32 1 0 overlap
```
