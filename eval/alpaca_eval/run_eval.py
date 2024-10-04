import os
import argparse
from datasets import load_dataset
from llm import LLMInferenceEngine, GenerationArgs, UniversalGenParams
from hf_ft import TULU_CHAT_TEMPLATE

ALPACA_EVAL_COMMAND_TEMPLATE = """
alpaca_eval evaluate \
--model_outputs "{alpaca_eval_input_file_path}" \
--annotators_config "weighted_alpaca_eval_gpt4_turbo" \
--output_path "{alpaca_eval_results_save_dir}" \
--name "{model_name}-greedy" \
--reference_outputs "{reference_path}"
""".strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--use_chat_format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference_path", type=str, default=None)
    parser.add_argument("--prompt_format", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_tulu_chat_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop", type=str, default=None)
    args = parser.parse_args()
    args.stop = args.stop.strip().split(",") if args.stop is not None else None
    if args.save_dir is None:
        args.save_dir = os.path.join(args.model, "evals/alpaca_eval") if os.path.exists(args.model) else os.path.join("outputs/remote_models", args.model.split("/")[-1], "evals/alpaca_eval")
    if args.reference_path is not None and ".json" not in args.reference_path:
        if os.path.exists(os.path.join(args.reference_path, "evals/alpaca_eval/model_outputs.json")):
            args.reference_path = os.path.join(args.reference_path, "evals/alpaca_eval/model_outputs.json") 
    return args

def stop_remover(text, stop):
    for s in stop:
        if text.rstrip().endswith(s):
            return text[:-len(s)].rstrip()
    return text

def main():
    args = parse_args()
    model_name = args.model.replace("/", "_")
    alpaca_eval_input_file_path = os.path.join(args.save_dir, "model_outputs.json")

    if not os.path.exists(alpaca_eval_input_file_path):
        if args.model_engine_backend == "vllm":
            backend_kwargs = {"tensor_parallel_size": args.model_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
        elif args.model_engine_backend == "vllm-openai":
            backend_kwargs = {"base_url": args.model_backend_base_url}
        else:
            backend_kwargs = {}
        model = LLMInferenceEngine(args.model, args.model_engine_backend, backend_kwargs=backend_kwargs, custom_chat_template=TULU_CHAT_TEMPLATE if args.enforce_tulu_chat_format else None)
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
        dataset = dataset.remove_columns(["output", "generator"])

        instructions = dataset["instruction"]
        if args.prompt_format is not None:
            instructions = [args.prompt_format.format(instruction=instruction) for instruction in instructions]

        os.makedirs(args.save_dir, exist_ok=True)
        prediction_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0, stop=args.stop)
        prediction_gen_args = GenerationArgs(
            engine_input=instructions,
            gen_params=prediction_gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            apply_chat_template=args.use_chat_format,
        )
        model_outputs = model.generate(prediction_gen_args)
        model_outputs = [output.output_seqs[0] for output in model_outputs]
        if args.stop is not None:
            model_outputs = [stop_remover(output, args.stop) for output in model_outputs]
        dataset = dataset.add_column(name="output", column=model_outputs)
        dataset = dataset.add_column(name="generator", column=[f"{model_name}-greedy"] * len(dataset))
        
        dataset.to_json(alpaca_eval_input_file_path, lines=False, indent=4)

    if args.reference_path is not None:
        alpaca_eval_results_save_dir = os.path.join(args.save_dir, f"vs_{args.reference_path.replace('/', '_').replace('.json', '')}")
    else:
        alpaca_eval_results_save_dir = os.path.join(args.save_dir, f"vs_default_reference")
    os.makedirs(alpaca_eval_results_save_dir, exist_ok=True)

    eval_command = ALPACA_EVAL_COMMAND_TEMPLATE.format(
        alpaca_eval_input_file_path=alpaca_eval_input_file_path,
        alpaca_eval_results_save_dir=alpaca_eval_results_save_dir,
        model_name=model_name,
        reference_path=args.reference_path,
    )

    if args.reference_path is None:
        eval_command = eval_command.split("--reference_outputs")[0].strip()

    script_save_path = os.path.join(alpaca_eval_results_save_dir, f"run_eval.sh")
    with open(script_save_path, "w") as f:
        f.write(eval_command)

if __name__ == "__main__":
    main()