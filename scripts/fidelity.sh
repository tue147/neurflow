#!/bin/bash

LABEL_LIST=({0..50})
MODEL="googlenet"

TAU_LIST=(4 8 16)
DATA_DIR="/mnt/disk1/user/Tue.CM210908/imagenet"
KEEP_DATA=False
BATCH_SIZE=128
BATCH_SIZE_IG=4
DEVICE="cuda:1"

cd ..

for TAU in "${TAU_LIST[@]}"; do
    for LABEL in "${LABEL_LIST[@]}"; do
        OUTPUT_DIR="./exps/fidelity_of_neuron/full_${TAU}"
        python run.py --label "$LABEL" --model "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --tau $TAU --keep_data "$KEEP_DATA" --batch_size "$BATCH_SIZE" --batch_size_ig "$BATCH_SIZE_IG" --device "$DEVICE"
    done
done

LOAD_DIR="./exps/fidelity_of_neuron/full_"
python -m exps.fidelity_of_neuron.Fidelity.py --data_dir "$DATA_DIR" --label_list "${LABEL_LIST[@]}" --model "$MODEL" --load_dir "$LOAD_DIR" --list_tau "${TAU_LIST[@]}" --batch_size "$BATCH_SIZE" --device "$DEVICE"