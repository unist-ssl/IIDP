# IIDP: Independent and Identical Data Parallelism
## Introduction
*IIDP* (*I*ndependent and *I*dentical *D*ata *P*arallelism) is a Deep Neural Network (DNN) training framework
which provides the same theoretical convergence rate of distributed SGD on heterogeneous GPUs.
In IIDP, each worker uses the same local batch size, but multiple workers called *virtual stream workers* (VSWs) can be executed concurrently on a single GPU by leveraging the GPU multi-stream.
IIDP also allows these VSWs to execute multiple mini-batches via Gradient Accumulation (GA).
Furthermore, IIDP employs an efficient synchronization mechanism among VSWs, that is, the local aggregation and weight update techniques.

For more details, please refer to EuroSys '25 paper entitled **JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs** (link will be uploaded).

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

Install Anaconda (Optional) - If Anaconda has already been installed, skip this step.
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

Prepare conda environment
```bash
CONDA_ENV=iidp
conda create -n $CONDA_ENV python=3.8 -y
conda activate $CONDA_ENV
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install -c pytorch magma-cuda111 -y # For CUDA 11.1
```
Install pytorch and vision by building source code
```bash
BASE=$HOME # Set the custom base path
IIDP_HOME=$BASE/IIDP
PYTORCH_HOME=$BASE/pytorch
VISION_HOME=$BASE/vision

cd $BASE

git clone https://github.com/unist-ssl/IIDP

git clone --recursive -b v1.8.1 https://github.com/pytorch/pytorch.git
cd $PYTORCH_HOME
patch -p1 < $IIDP_HOME/pytorch.patch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

cd $BASE
git clone -b v0.9.1 https://github.com/pytorch/vision.git
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
