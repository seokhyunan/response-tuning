import json
import numpy as np
import re
import argparse
import os

REGULAR_CRITERIA = ["helpfulness", "factuality",  "clarity", "depth", "engagement"]
SAFETY_CRITERIA = ["safety"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regular_eval_output_path", type=str, default=None)
    parser.add_argument("--safety_eval_output_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.regular_eval_output_path)
    return args


def extract_int(input_string):
    int_list = re.findall(r'\d+', input_string)
    int_list = [int(num) for num in int_list]
    return int_list[-1] if len(int_list) > 0 else None


def compute_metrics(regular_eval_results, safety_eval_results):
    metrics = {f"avg_{criteria}_score": np.mean([result[criteria]["score"] for result in regular_eval_results if result is not None]) for criteria in REGULAR_CRITERIA}
    if safety_eval_results is not None:
        metrics.update({f"avg_{criteria}_score": np.mean([result[criteria]["score"] for result in safety_eval_results if result is not None]) for criteria in SAFETY_CRITERIA})
        metrics["avg_score"] = np.mean([metrics[f"avg_{criteria}_score"] for criteria in REGULAR_CRITERIA + SAFETY_CRITERIA])
    else:
        metrics["avg_score"] = np.mean([metrics[f"avg_{criteria}_score"] for criteria in REGULAR_CRITERIA])
    return metrics

def criteria_generation_matcher(criteria, eval_output_criteria):
    for g in eval_output_criteria:
        if g.startswith(criteria):
            return criteria, g
    return None

def eval_output_processor(eval_output):
    if "safety" in eval_output: # safety eval
        for criteria in REGULAR_CRITERIA:
            eval_output[criteria] = {"reason": None, "score": None}
    else:
        for criteria in SAFETY_CRITERIA: # regular eval
            eval_output[criteria] = {"reason": None, "score": None}

    for criteria in REGULAR_CRITERIA + SAFETY_CRITERIA:
        eval_element = eval_output.get(criteria, None)
        if eval_element is None:
            c, gc = criteria_generation_matcher(criteria, eval_output.keys())
            eval_output[c] = eval_output.pop(gc)
        if isinstance(eval_output[criteria]["score"], str):
            eval_output[criteria]["score"] = extract_int(eval_output[criteria]["score"])

    return eval_output

def just_eval_output_processor(regular_eval_output, safety_eval_output):
    regular_eval_output = json.load(open(regular_eval_output, "r"))
    regular_ids = sorted([output["id"] for output in regular_eval_output])

    # regular eval outputs
    id_eval_results_map = dict.fromkeys(regular_ids)
    for eval_output in regular_eval_output:
        validity = eval_output.get("parsed_result", None)
        if validity:
            id_eval_results_map[eval_output["id"]] = eval_output_processor(eval_output["parsed_result"])
        else:
            id_eval_results_map[eval_output["id"]] = None
    regular_eval_results = [id_eval_results_map[id] for id in regular_ids]

    if safety_eval_output is not None:
        safety_eval_output = json.load(open(safety_eval_output, "r"))
        safety_ids = sorted([output["id"] for output in safety_eval_output])
        # safety eval outputs
        id_eval_results_map = dict.fromkeys(safety_ids)
        for eval_output in safety_eval_output:
            id_eval_results_map[eval_output["id"]] = eval_output_processor(eval_output["parsed_result"])
        safety_eval_results = [id_eval_results_map[id] for id in safety_ids]
    else:
        safety_eval_results = None
        
    return regular_eval_results, safety_eval_results

def main():
    args = parse_args()
    regular_eval_results, safety_eval_results = just_eval_output_processor(args.regular_eval_output_path, args.safety_eval_output_path)
    total_len = len(regular_eval_results) if safety_eval_results is None else len(regular_eval_results) + len(safety_eval_results)
    metrics = compute_metrics(regular_eval_results, safety_eval_results)
    
    with open(os.path.join(args.save_dir, f"metrics_{total_len}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model name: {args.regular_eval_output_path}")
    for criteria in REGULAR_CRITERIA:
        print(f"{criteria}: {metrics[f'avg_{criteria}_score']}")
    if safety_eval_results is not None:
        for criteria in SAFETY_CRITERIA:
            print(f"{criteria}: {metrics[f'avg_{criteria}_score']}")

if __name__ == "__main__":
    main()