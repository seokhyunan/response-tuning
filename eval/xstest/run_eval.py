import argparse
from llm import LLMInferenceEngine, GenerationArgs, UniversalGenParams
from datasets import load_dataset
import random
import os
import json
from eval.safety_eval_utils import REFUSAL_JUDGE_PROMPT, refusal_judge_output_parser, compute_exaggerate_safety_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=int, default=None)
    parser.add_argument("--evaluator", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--evaluator_engine_backend", type=str, default="vllm-openai")
    parser.add_argument("--evaluator_backend_base_url", type=int, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--evaluator_num_gpus", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_chat_format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(args.model, "evals/xstest") if os.path.exists(args.model) else os.path.join("outputs/remote_models", args.model.split("/")[-1], "evals/xstest")
    return args

def load_and_reformat():
    return load_dataset("walledai/XSTest")["test"]

def main():
    args = parse_args()
    if args.model_engine_backend == "vllm":
        backend_kwargs = {"tensor_parallel_size": args.model_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
    elif args.model_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.model_backend_base_url}
    else:
        backend_kwargs = {}
    model = LLMInferenceEngine(args.model, backend=args.model_engine_backend, backend_kwargs=backend_kwargs)
    dataset = load_and_reformat()

    if args.max_num_examples is not None:
        random.seed(0)
        random_ids = sorted(random.sample(range(len(dataset)), min(args.max_num_examples, len(dataset))))
        dataset = dataset.select(random_ids)

    prediction_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
    prediction_gen_args = GenerationArgs(
        engine_input=dataset["prompt"],
        gen_params=prediction_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
        apply_chat_template=args.use_chat_format,
    )

    model_outputs = model.generate(prediction_gen_args)
    model_outputs = [output.output_seqs[0] for output in model_outputs]
    dataset = dataset.add_column(name="model_output", column=model_outputs)

    # del model
    if args.evaluator_engine_backend == "vllm":
        backend_kwargs = {"tensor_parallel_size": args.evaluator_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
    elif args.evaluator_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.evaluator_backend_base_url}
    else:
        backend_kwargs = {}
    evaluator = LLMInferenceEngine(args.evaluator, backend=args.evaluator_engine_backend, backend_kwargs=backend_kwargs)
    eval_inputs = [REFUSAL_JUDGE_PROMPT.format(prompt=example["prompt"], model_output=example["model_output"]) for example in dataset]
    eval_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
    eval_gen_args = GenerationArgs(
        engine_input=eval_inputs,
        gen_params=eval_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
        apply_chat_template=True,
    )
    eval_outputs = evaluator.generate(eval_gen_args)
    eval_outputs = [output.output_seqs[0] for output in eval_outputs]
    eval_labels = [refusal_judge_output_parser(output) for output in eval_outputs]
    dataset = dataset.add_column(name="refusal_clf_output", column=eval_outputs)
    dataset = dataset.add_column(name="refusal_clf_label", column=eval_labels)
    dataset = dataset.add_column(name="refusal_clf", column=[args.evaluator] * len(dataset))

    metrics = compute_exaggerate_safety_metrics(dataset, example_type_key="label")

    if args.max_num_examples is None:
        results_save_path = os.path.join(args.save_dir, f"results_xstest_evaluator_{args.evaluator.split('/')[-1]}.jsonl")
        metrics_save_path = os.path.join(args.save_dir, f"metrics_xstest_evaluator_{args.evaluator.split('/')[-1]}.json")
    else:
        results_save_path = os.path.join(args.save_dir, f"results_xstest_{args.max_num_examples}_evaluator_{args.evaluator.split('/')[-1]}.jsonl")
        metrics_save_path = os.path.join(args.save_dir, f"metrics_xstest_{args.max_num_examples}_evaluator_{args.evaluator.split('/')[-1]}.json")

    dataset.to_json(results_save_path, lines=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()