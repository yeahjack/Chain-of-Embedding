#!/bin/bash

export PROJECT_PATH="/your/path/to/Chain-of-Embedding"
export CUDA_VISIBLE_DEVICES="0,1"

model_name="qwen2-7B-Instruct"
dataset_list=(mgsm)

for i in ${dataset_list[*]}; do
    python main.py --model_name $model_name \
                        --dataset "$i" \
                        --print_model_parameter \
                        --save_output \
                        --save_hidden_states \
                        --save_coe_score \
                        --save_coe_figure
done
