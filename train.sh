#! /bin/bash

set -x

python main.py \
--dataset_dir datasets \
--dataset CUB \
--attr_subset cbm \
--use_class_level_attr