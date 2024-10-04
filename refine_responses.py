import argparse
from llm import LLMInferenceEngine, GenerationArgs, UniversalGenParams
from datasets import Dataset
import re

RESPONSE_REFINE_PROMPT_TEMPLATE = """
Your task is to refine and enhance the response of an AI chat assistant. The goal is to make the response more clear, well-structured, and engaging. You will be provided with the user request and the corresponding response. Revise the response, focusing on the following aspects:

1. Clarity: Make the response easy to understand. It should be direct and to the point, avoiding complex language that might confuse the user.
2. Structure: Organize the content in a logical and coherent manner. The response should flow naturally, making it easy for the user to follow along and grasp the key points.
3. Tone: Adjust the tone to be friendly, conversational, and engaging. The response should feel approachable and enjoyable, as if having a pleasant conversation with the user.

Steps for Refinement:
1. Begin by briefly reviewing the response and identifying areas that could be improved.
2. Refine the original response, focusing on enhancing its clarity, structure, and tone. Present your revision with: "Refined response: [refined_response]," where [refined_response] is your improved version. Do not include any additional explanations after "Refined response:".

Now, please refine the following response:

<BEGIN USER REQUEST>{user_request}<END USER REQUEST>
<BEGIN ASSISTANT RESPONSE>{response}<END ASSISTANT RESPONSE>
""".strip()

def extract_revision(refiner_output, original, revision_start_str):
    # Define the pattern based on the provided pattern type
    pattern = rf"{revision_start_str}\s*(.*)"
    # "refined response:" or "refined text:"

    # Search for the pattern in the refiner output, ignoring case
    match = re.search(pattern, refiner_output, re.IGNORECASE | re.DOTALL)

    if match:
        # Extract the refined data (response or text) from the match object
        refined_data = match.group(1).strip()
        return refined_data

    # Return the original if no match is found or if it's identical to the original
    return original

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refiner', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--refiner_engine_backend', type=str, default="vllm")
    parser.add_argument("--refiner_base_url", type=int, default=None)
    parser.add_argument('--refiner_num_gpus', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default='datasets/train/lima.jsonl')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.refiner_engine_backend == "vllm":
        backend_kwargs = {"tensor_parallel_size": args.refiner_num_gpus, "gpu_memory_utilization": args.gpu_memory_utilization}
    elif args.refiner_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.refiner_base_url}
    else:
        backend_kwargs = {}
    refiner = LLMInferenceEngine(args.refiner, args.refiner_engine_backend, backend_kwargs=backend_kwargs)
    dataset = Dataset.from_json(args.dataset_path)

    revision_prefix = "refined response:" 
    refine_prompts = [RESPONSE_REFINE_PROMPT_TEMPLATE.format(user_request=example["prompt"], response=example["completion"]) for example in dataset]

    refiner_gen_params = UniversalGenParams(n=1, max_new_tokens=8192, temperature=0)
    refiner_gen_args = GenerationArgs(
        engine_input=refine_prompts,
        gen_params=refiner_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
        apply_chat_template=True,
    )
    refiner_outputs = refiner.generate(refiner_gen_args)
    refined_responses = [extract_revision(refiner_output.output_seqs[0], example["completion"], revision_prefix) for refiner_output, example in zip(refiner_outputs, dataset)]

    dataset = dataset.rename_column("completion", "source_completion")
    dataset = dataset.add_column("completion", refined_responses)
    save_path = args.dataset_path.replace(".jsonl", f"_response_refined_by_{args.refiner.split('/')[-1]}_V4.jsonl")
    dataset.to_json(save_path, lines=True)

    # check refine extraction success rate
    num_refined_responses = len([1 for example in dataset if example["source_completion"] != example["completion"]])
    extraction_rate = num_refined_responses / len(refined_responses) * 100
    print(f"Refined responses extracted successfully for {num_refined_responses} out of {len(refined_responses)} examples ({extraction_rate:.2f}%).")

if __name__ == "__main__":
    main()