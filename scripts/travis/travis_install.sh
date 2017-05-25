#!/bin/bash
# This script must be run with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"

# Install apt packages where the Ubuntu 12.04 default and ppa works for Caffe

# This ppa is for gflags and glog
apt-get -y update
apt-get install -y --no-install-recommends \
    build-essential \
    wget git curl \
    python-dev python-numpy \
    libleveldb-dev libsnappy-dev libopencv-dev \
    libboost-dev libboost-system-dev libboost-python-dev libboost-thread-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev libgflags-dev libgoogle-glog-dev \
    bc

if $WITH_CMAKE; then
  apt-get -y --no-install-recommends install cmake
fi

# Install CUDA, if needed
if $WITH_CUDA; then
  # install repo packages
  CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG

  if $WITH_CUDNN ; then
    ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
    dpkg -i $ML_REPO_PKG
  fi

  # update package lists
  apt-get -y update

  # install packages
  CUDA_PKG_VERSION="7-5"
  CUDA_VERSION="7.5"
  apt-get install -y --no-install-recommends \
    cuda-core-$CUDA_PKG_VERSION \
    cuda-cudart-dev-$CUDA_PKG_VERSION \
    cuda-cublas-dev-$CUDA_PKG_VERSION \
    cuda-curand-dev-$CUDA_PKG_VERSION
  # manually create CUDA symlink
  ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

  if $WITH_CUDNN ; then
    apt-get install -y --no-install-recommends libcudnn6-dev
  fi
fi

# Install LMDB
apt-get install -y --no-install-recommends \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libsnappy-dev

# Install the Python runtime dependencies via miniconda (this is much faster
# than using pip for everything).
if [ -d "$HOME/miniconda" ]; then
  # Control will enter here if $DIRECTORY exists.
  echo "Removing previous miniconda"
  rm -rf $HOME/miniconda
fi

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --yes conda
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf

# Install OpenMPI 1.
