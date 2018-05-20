#!/usr/bin/env bash

# install additional depts
sudo apt install python-pip python-dev unzip python-matplotlib
sudo pip install cython scikit-image easydict opencv-python

# build cython extension
make
