#! /bin/bash

set -x

python main.py \
--dataset_dir datasets \
--dataset CUB \
--attr_subset cbm \
--backbone resnet101 \
--use_class_level_attr 