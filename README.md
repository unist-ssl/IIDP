# IIDP: Independent and Identical Data Parallelism
## Introduction
*IIDP* (*I*ndependent and *I*dentical *D*ata *P*arallelism) is a Deep Neural Network (DNN) training framework
which provides the same theoretical convergence rate of distributed SGD on heterogeneous GPUs.
In IIDP, each worker uses the same local batch size, but multiple workers called *virtual stream workers* (VSWs) can be executed concurrently on a single GPU by leveraging the GPU multi-stream.
IIDP also allows these VSWs to execute multiple mini-batches via Gradient Accumulation (GA).
Furthermore, IIDP employs an efficient synchronization mechanism among VSWs, that is, the local aggregation and weight update techniques.
For more details, please refer to EuroSys '25 paper (link will be uploaded).

## Getting Started
### Prerequisites
* Ubuntu >= 16.04
* Anaconda3 4.13.0
* Python 3.8
* NVIDIA driver >= 450.80.02
* CUDA 11.1
* cuDNN 8.2.1

### Software Packages Installation
Install CUDA and CuDNN
- CUDA download toolkit [[link]](https://developer.nvidia.com/cuda-toolkit-archive). Make sure that `/usr/local/cuda` is linked to `/usr/local/cuda-11.1`.
- CuDNN download toolikt [[link]](https://developer.nvidia.com/rdp/cudnn-archive).

Prepare conda environment
```bash
conda create -n iidp python=3.8 -y
conda activate iidp
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install -c pytorch magma-cuda111 -y # For CUDA 11.1
```
Install pytorch and vision by building source code
```bash
cd $HOME
# TODO: git clone ${repo url}
IIDP_HOME=$HOME/IIDP

git clone --recursive -b v1.8.1 https://github.com/pytorch/pytorch.git
PYTORCH_HOME=$HOME/pytorch
cd $PYTORCH_HOME
patch -p1 < $IIDP_HOME/pytorch.patch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

cd $HOME
git clone -b v0.9.1 https://github.com/pytorch/vision.git
VISION_HOME=$HOME/vision
cd $VISION_HOME
pip install pillow==10.4.0
python setup.py install
```
Install IIDP
```bash
cd $IIDP_HOME
pip install -r requirements.txt
python setup.py install
```

## QuickStart for ResNet-50
Refer to [README.md](examples/imagenet/resnet50_quickstart/) in ```examples/imagenet/resnet50_quickstart/``` directory.

## Run IIDP
Refer to [README.md](examples/) in ```examples/``` directory.
