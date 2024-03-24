#! /bin/bash

set -x

mkdir -p datasets && cd datasets && mkdir -p CUB
wget -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -zxf CUB_200_2011.tgz -C CUB --no-same-owner
