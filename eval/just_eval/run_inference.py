import argparse
import os
import random
from llm import LLMInferenceEngine, GenerationArgs, UniversalGenParams
from datasets import load_dataset

JUST_EVAL_COMMAND_TEMPLATE = """
just_eval \
    --mode "{mode}" \
    --gpt_model "gpt-4o-2024-08-06" \
    --model_output_file {temp_output_path} \
    --eval_output_file {eval_output_path}
""".strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--use_chat_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--rstrip", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    args.stop = args.stop.strip().split(",") if args.stop is not None else None
    if args.save_dir is None:
        args.save_dir = os.path.join(args.model, "evals/just_eval") if os.path.exists(args.model) else os.path.join("outputs/remote_models", args.model.split("/")[-1], "evals/just_eval")
        if args.prompt is not None:
            prompt_name = os.path.splitext(os.path.basename(args.prompt))[0]
            args.save_dir = os.path.join(args.save_dir, prompt_name)
    return args

def load_and_reformat():
    test_data = load_dataset("re-align/just-eval-instruct")["test"]
    test_data = test_data.sort("id")
    return test_data

def stop_remover(text, stop):
    for s in stop:
        if text.rstrip().endswith(s):
            return text[:-len(s)].rstrip()
    return text

def main():
    args = parse_args()
    dataset = load_and_reformat()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.max_num_examples is not None:
        random.seed(0)
        random_ids = sorted(random.sample(range(len(dataset)), min(args.max_num_examples, len(dataset))))
        dataset = dataset.select(random_ids)
        just_eval_regular_input_path = os.path.join(args.save_dir, f"just_eval_input_regular_model_outputs_{args.max_num_examples}.json")
        just_eval_safety_input_path = os.path.join(args.save_dir, f"just_eval_input_safety_model_outputs_{args.max_num_examples}.json")
    else:
        just_eval_regular_input_path = os.path.join(args.save_dir, "just_eval_input_regular_model_outputs.json")
        just_eval_safety_input_path = os.path.join(args.save_dir, "just_eval_input_safety_model_outputs.json")

    if not os.path.exists(just_eval_regular_input_path) or not os.path.exists(just_eval_safety_input_path):
        if args.model_engine_backend == "vllm":
            backend_kwargs = {"tensor_parallel_size": args.model_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
        elif args.model_engine_backend == "vllm-openai":
            backend_kwargs = {"base_url": args.model_backend_base_url}
        else:
            backend_kwargs = {}
        model = LLMInferenceEngine(args.model, args.model_engine_backend, backend_kwargs=backend_kwargs)
        instructions = dataset["instruction"]
        if args.prompt is not None:
            prompt = open(args.prompt, "r", encoding='utf-8').read().strip()
            instructions = [prompt.format(instruction=instruction) for instruction in instructions]

        prediction_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0, stop=args.stop)
        prediction_gen_args = GenerationArgs(
            engine_input=instructions,
            gen_params=prediction_gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            apply_chat_template=args.use_chat_format
        )

        model_outputs = model.generate(prediction_gen_args)
        model_outputs = [output.output_seqs[0] for output in model_outputs]
        if args.stop is not None:
            model_outputs = [stop_remover(output, args.stop).strip() for output in model_outputs]
        if args.rstrip is not None:
            model_outputs = [output.rstrip(args.rstrip) for output in model_outputs]
        dataset = dataset.add_column(name="input_prompt", column=instructions)
        dataset = dataset.remove_columns("output").add_column(name="output", column=model_outputs)
        dataset = dataset.remove_columns("generator").add_column(name="generator", column=[args.model]*len(dataset))

        regular, safety = dataset.filter(lambda x: x["category"] == "regular"), dataset.filter(lambda x: x["category"] == "safety")
        regular.to_json(just_eval_regular_input_path, lines=False)
        safety.to_json(just_eval_safety_input_path, lines=False)

    regular_eval_output_path = os.path.join(args.save_dir, "just_eval_regular_eval_output.json")
    safety_eval_output_path = os.path.join(args.save_dir, "just_eval_safety_eval_output.json")
    
    regular_eval_command = JUST_EVAL_COMMAND_TEMPLATE.format(mode="score_multi", temp_output_path=just_eval_regular_input_path, eval_output_path=regular_eval_output_path)
    safety_eval_command = JUST_EVAL_COMMAND_TEMPLATE.format(mode="score_safety", temp_output_path=just_eval_safety_input_path, eval_output_path=safety_eval_output_path)

    # save eval command as .sh file
    with open(os.path.join(args.save_dir, "regular_eval_command.sh"), "w") as f:
        f.write(regular_eval_command)
    with open(os.path.join(args.save_dir, "safety_eval_command.sh"), "w") as f:
        f.write(safety_eval_command)

if __name__ == "__main__":
    main()