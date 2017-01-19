#!/usr/bin/env bash

# install additional depts
sudo apt install python-pip python-dev unzip
sudo pip install cython python-matplotlib python-skimage easydict

# install a forked MXNet
git clone https://github.com/precedenceguo/mxnet.git --recursive -b simple
pushd mxnet
cp make/config.mk ./
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
make -j$(nproc)
pushd python
python setup.py install --user
popd
popd

# build cython extension
make
