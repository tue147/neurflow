#!/bin/bash

LABEL_LIST=(0 1 2 3 4 5 6 7 8 9)
TAU_LIST=(10 30 50)
MODELS=("googlenet" "resnet50")


OUTPUT_DIR=".exps/optimality_of_neuron/run_each_layer"
DATA_DIR="/mnt/disk1/user/Tue.CM210908/imagenet"
BATCH_SIZE=128
BATCH_SIZE_IG=4
DEVICE="cuda:3"

cd ..

for MODEL in "${MODELS[@]}"; do
    for TAU in "${TAU_LIST[@]}"; do
        for LABEL in "${LABEL_LIST[@]}"; do
            python -m exps.optimality_of_neuron.run_each_layer.py --label "$LABEL" --model "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --tau $TAU --batch_size "$BATCH_SIZE" --batch_size_ig "$BATCH_SIZE_IG" --device "$DEVICE"
        done
    done
done

LOAD_DIR="$OUTPUT_DIR"
MINIMIZATION_DIR=".exps/optimality_of_neuron/minimization"
NUM_NODE_TEST=5
NUM_RANDOM_COMB=100
BATCH_SIZE_MINIMIZATION=1024

for MODEL in "${MODELS[@]}"; do
    for TAU in "${TAU_LIST[@]}"; do
        for LABEL in "${LABEL_LIST[@]}"; do
            python -m exps.optimality_of_neuron.Minimization.py --data_dir "$DATA_DIR" --label "$LABEL" --model "$MODEL" --load_dir "$LOAD_DIR" --tau $TAU --batch_size "$BATCH_SIZE" --num_node_test "$NUM_NODE_TEST" --num_random_comb "$NUM_RANDOM_COMB" --output_dir "$MINIMIZATION_DIR" --device "$DEVICE"
        done
    done
done