#!/usr/bin/env bash

pushd data

# the result is selective_search_data
wget http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
tar xf selective_search_data.tgz

popd
