#!/bin/bash

LABEL=309
MODEL="resnet50"
OUTPUT_DIR=".exps/compare/results_neucept"
DATA_DIR="/mnt/disk1/user/Tue.CM210908/imagenet"
TAU=16
KEEP_DATA=True
BATCH_SIZE=128
BATCH_SIZE_NEUCEPT=50
DEVICE="cuda:1"

cd ..

python -m exps.compare.run_with_neucept.py --label "$LABEL" --model "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --tau $TAU --keep_data "$KEEP_DATA" --batch_size "$BATCH_SIZE" --batch_size_neucept "$BATCH_SIZE_NEUCEPT" --device "$DEVICE"