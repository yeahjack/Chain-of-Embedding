#!/bin/bash

export PROJECT_PATH="/mnt/nas/users/jiangchuan.wym/chain-of-embedding"
export CUDA_VISIBLE_DEVICES="2,3"

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