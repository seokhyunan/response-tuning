from datasets import load_dataset
import argparse
import random
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lima')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--max_num_examples", type=int, default=None, help="cut dataset to a certain size")
    return parser.parse_args()

def random_sampler(dataset, num_samples):
    random.seed(0)
    sampling_ids = list(range(len(dataset)))
    random.shuffle(sampling_ids)
    return dataset.select(sampling_ids[:num_samples])

def lima_processor(example):
    return {"prompt": example["conversations"][0], "completion": example["conversations"][1]}

def alpaca_processor(example, instruction_key="instruction", response_key="output", input_key="input"):
    return {"prompt": example[instruction_key] if not example[input_key] else f"{example[instruction_key]}\n\n{example[input_key]}", "completion": example[response_key]}

def load_and_reformat(dataset_path):
    if dataset_path == "lima":
        dataset = load_dataset('GAIR/lima')['train'].select(list(range(1000))).map(lima_processor)

    elif dataset_path == 'alpaca':
        dataset = load_dataset('yahma/alpaca-cleaned')['train'].map(alpaca_processor, num_proc=64)

    elif dataset_path == 'dolly':
        dataset = load_dataset('databricks/databricks-dolly-15k')['train']
        dataset = dataset.map(
            alpaca_processor,
            fn_kwargs={"instruction_key": "instruction", "response_key": "response", "input_key": "context"},
        )
    else:
        raise NotImplementedError

    return dataset

def main():
    args = parse_args()
    dataset = load_and_reformat(args.dataset)

    if args.max_num_examples is not None and args.max_num_examples < len(dataset):
        random.seed(0)
        index_shuffle = list(range(len(dataset)))
        random.shuffle(index_shuffle)
        dataset = dataset.select(index_shuffle[:args.max_num_examples])

    save_dir = os.path.join('datasets', 'train')
    if args.max_num_examples is not None:
        save_path = os.path.join(save_dir, f"{args.dataset}_{args.max_num_examples}.jsonl")
    else:
        save_path = os.path.join(save_dir, f"{args.dataset}.jsonl")
    dataset.to_json(save_path, lines=True)

if __name__ == '__main__':
    main()