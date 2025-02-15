#!/bin/bash

METHOD_NAME=("integrated_gradients" "smoothgrad" "guided_backprop" "gradient_shap" "knockoff" "lrp" "saliency")
MODEL_LIST=("resnet50" "googlenet")
LABEL_LIST=({0..10})
DATA_DIR="/mnt/disk1/user/Tue.CM210908/imagenet"
BATCH_SIZE=1024
LIST_TAU=(1 5 10 20 50)
NUM_COMB=500
OUTPUT_DIR="./exps/fidelity_of_weight/correlation/"
DEVICE="cuda:0"

cd ..

for METHOD in "${METHOD_NAME[@]}"; do
    for MODEL in "${MODEL_LIST[@]}"; do
        python -m exps.fidelity_of_weight.Correlation.py --method_name "$METHOD" --label_list "${LABEL_LIST[@]}" --model_name "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --batch_size "$BATCH_SIZE" --device "$DEVICE" --list_tau "${LIST_TAU[@]}" --num_comb "$NUM_COMB"
    done
done