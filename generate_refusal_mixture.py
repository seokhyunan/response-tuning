from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
import random
import os
from tqdm import tqdm
from generate_training_data import alpaca_processor, random_sampler
import json
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='datasets/train/alpaca.jsonl')
    parser.add_argument("--safety_examples_path", type=str, default="datasets/train/safety_only_data_Instructions.json")
    parser.add_argument("--num_safety_examples", type=int, default=100)
    return parser.parse_args()

def contextual_filter(example):
    extracted = re.findall(r"\b(ai|sorry)\b", example["completion"].strip().lower())
    return len(extracted) > 0

def mix(target_dataset_path, safety_examples_path, num_safety_examples):
    dataset = Dataset.from_json(target_dataset_path)
    safety_instructions = Dataset.from_json(safety_examples_path)
    safety_instructions = safety_instructions.map(
        alpaca_processor,
        remove_columns=safety_instructions.column_names,
        fn_kwargs={"instruction_key": "instruction", "response_key": "output", "input_key": "input"},
    )
    safety_instructions = safety_instructions.filter(contextual_filter)
    safety_instructions = random_sampler(safety_instructions, num_safety_examples)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in safety_instructions.column_names])
    return concatenate_datasets([dataset, safety_instructions])

def main():
    args = parse_args()
    dataset = mix(args.target_dataset, args.safety_examples_path, args.num_safety_examples)
    save_dir = os.path.join('datasets', 'train')
    dataset_name = os.path.splitext(os.path.basename(args.target_dataset))[0]
    save_path = os.path.join(save_dir, f"{dataset_name}+safety{args.num_safety_examples}.jsonl")
    dataset.to_json(save_path, lines=True)

if __name__ == '__main__':
    main()