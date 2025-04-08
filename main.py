import argparse
import os
import sys

import torch
from transformers import (
    GenerationConfig,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

project_root_path = os.environ["PROJECT_PATH"]
sys.path.append(project_root_path)
from config_pool import DATASET_POOL, LANGUAGE_MAPPING, MODEL_POOL
from Data.load_data import DatasetInfo
from inference import Inference
from Model.load_model import load_base_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="chain-of-embedding")

    parser.add_argument("--model_name", type=str, default="Llama-3-8B-Instruct", choices=MODEL_POOL)
    parser.add_argument("--dataset", type=str, default="mgsm", choices=DATASET_POOL)
    parser.add_argument("--max_output_token", type=int, default=2048)

    parser.add_argument("--print_model_parameter", action="store_true")
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--save_hidden_states", action="store_true")
    parser.add_argument("--save_coe_score", action="store_true")
    parser.add_argument("--save_coe_figure", action="store_true")

    args = parser.parse_args()
    args.max_output_token = 2048 if "Instruct" in args.model_name else 128

    model, tokenizer, config = load_base_model(args)
    if args.print_model_parameter:
        print("********** Module Name and Size **********\n")
        for param_tensor in model.state_dict():
            print(param_tensor,'\t',model.state_dict()[param_tensor].size())

    model_info = {
        "model_name": args.model_name,
        "model_ckpt": model,
        "tokenizer": tokenizer,
        "model_config": config,
        "generation_config": GenerationConfig(),
        "max_output_token": args.max_output_token
    }
    dataset_info = {
        "dataset_name": args.dataset,
    }
    verbose = {
        "save_output": args.save_output,
        "save_hidden_states": args.save_hidden_states,
        "save_coe_score": args.save_coe_score,
        "save_coe_figure": args.save_coe_figure
    }

    print(f"***** Model Name: *****\n{args.model_name}")
    print(f"***** Dataset Name: *****\n{args.dataset}")
    print(f"***** Dataset Size: *****\n{DatasetInfo(args.dataset).data_size}")

    language_list = LANGUAGE_MAPPING[args.dataset] if args.dataset in LANGUAGE_MAPPING else ["en"]
    for lang in language_list:
        dataset_info["language"] = lang
        Infer = Inference(model_info, dataset_info, verbose)
        Infer.dataset_inference()
