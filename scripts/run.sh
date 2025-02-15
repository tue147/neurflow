#!/bin/bash

LABEL=309
MODEL="resnet50"
OUTPUT_DIR="./results"
DATA_DIR="/mnt/disk1/user/Tue.CM210908/imagenet"
TAU=16
KEEP_DATA=True
BATCH_SIZE=128
BATCH_SIZE_IG=4
DEVICE="cuda:0"

cd ..

python run.py --label "$LABEL" --model "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --tau $TAU --keep_data "$KEEP_DATA" --batch_size "$BATCH_SIZE" --batch_size_ig "$BATCH_SIZE_IG" --device "$DEVICE"