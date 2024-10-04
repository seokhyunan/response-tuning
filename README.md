# Response Tuning: Aligning Large Language Models without Instructions

This repository contains the code and resources supporting the paper ["Response Tuning: Aligning Large Language Models without Instructions"](https://arxiv.org/abs/2410.02465) by Seokhyun An and Hyounghun Kim.

## Overview

Our paper presents Response Tuning (RT), which eliminates the instruction-conditioning step in instruction tuning and focuses solely on response supervision. By omitting the instruction-conditioning step, we can examine the isolated impact of establishing an adequate response space in alignment. We provide the training and evaluation scripts to aid reproducibility.

In the paper, we have shown that:

- RT models, trained solely using responses, can effectively respond to a wide range of instructions. They exhibit comparable performance to instruction-tuned models across various benchmarks, including AlpacaEval, JustEval, MMLU, PIQA, and more.
- By controlling the training response distribution, we can achieve specific alignment objectives:
    - Refining the structural attributes of responses can improve user preference.
    - Incorporating refusals into the RT data enables the models to responsibly handle unsafe queries.
- These observations also hold in an in-context learning setting.

## Prerequisites

```bash
conda create -n rt python=3.11
conda activate rt
pip install -r requirements.txt
./prepare_data.sh
```

<details>
<summary>Note on JustEval Benchmark</summary>

For the JustEval Benchmark, we found that its dependencies conflict with our experimental environment. Therefore, please create a separate environment to run JustEval on the trained model:

```bash
conda create -n just_eval python=3.11
conda activate just_eval
pip install git+https://github.com/Re-Align/just-eval.git
```
</details>

## Training Data Preparation

We currently support the Alpaca, Dolly, and LIMA datasets. Reformat and save the training datasets using the following script:

For Instruction Tuning (IT) and RT:

```bash
python3 generate_training_data.py --dataset [lima/alpaca/dolly]
```

To reproduce our results from the response refinement experiments, refine the datasets using the script below and train the model using the refined dataset:

```bash
CUDA_VISIBLE_DEVICES=[GPU_IDS] python3 refine_responses.py \
    --dataset_path [REFINEMENT_TARGET_DATASET_PATH] \
    --refiner [HF_MODEL_PATH]
```

<details>
<summary>Example</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 refine_responses.py \
    --dataset_path "datasets/train/lima.jsonl" \
    --refiner "meta-llama/Meta-Llama-3.1-70B-Instruct"
```
</details>

To reproduce our safety experiments, create an IT dataset mixed with safety-focused examples, and train the model using this dataset:


```bash
python3 generate_safety_mixture.py \
    --target_dataset [TRAINING_DATASET_PATH] \
    --num_safety_examples [NUMBER_OF_SAFETY_EXAMPLES_TO_MIX]
```

<details>
<summary>Example</summary>

```bash
python3 generate_safety_mixture.py \
    --target_dataset "datasets/train/lima.jsonl" \
    --num_safety_examples 200
```
</details>



## Model Training (QLoRA)

You can perform both IT and RT using the following scripts.

For IT:

```bash
./scripts/train/train_it.sh [GPU_IDS] [HF_MODEL_PATH] [DATASET_PATH]
```

<details>
<summary>Example</summary>

```bash
./scripts/train/train_it.sh 0 "meta-llama/Llama-3.1-8B" "datasets/train/lima.jsonl"
```
</details>

For RT:

```bash
./scripts/train/train_rt.sh [GPU_IDS] [HF_MODEL_PATH] [DATASET_PATH]
```

<details>
<summary>Example</summary>

```bash
./scripts/train/train_rt.sh 0 "meta-llama/Llama-3.1-8B" "datasets/train/lima.jsonl"
```
</details>

After training, you can interact with the trained model:
```bash
CUDA_VISIBLE_DEVICES=[GPU_IDS] python3 chat.py --model_id [MODEL_PATH]
```


## Instructability Evaluation

You can perform AlpacaEval, JustEval, and core capabilities evaluations (MMLU, OpenBookQA, HellaSwag, ARC, GSM8K, and PIQA) using the following scripts. After running the evaluation, you can find the results in `[ckpt_path]/evals/[benchmark_name]`.

For AlpacaEval:

Run the following script and then execute `run_eval.sh` generated in the `[ckpt_path]/evals/alpaca_eval` directory.

```bash
./scripts/eval/alpaca_eval.sh [GPU_IDS] [MODEL_PATH] [REFERENCE_OUTPUT_PATH (optional)]
```

For JustEval:

Run the following script and then execute `run_eval.sh` generated in the `[ckpt_path]/evals/justeval` directory. After running the evaluation script, compute the metrics.

```bash
./scripts/eval/just_eval/just_eval.sh [GPU_IDS] [MODEL_PATH]

# After running run_eval.sh
scripts/eval/just_eval/just_eval_compute_metric.sh [MODEL_PATH]
```

For core capabilities evaluation:

```bash
./scripts/eval/core.sh [GPU_IDS] [MODEL_PATH]
```

<details>
<summary>Note on CUDA Out-of-Memory (OOM) errors</summary>

When using Mistral and Gemma-2 models with vLLM, you may encounter CUDA OOM errors. To mitigate this issue, reduce the `gpu_memory_utilization` parameter by setting it to a lower value. You can do this by uncommenting the following lines in the evaluation script:

```bash
# if [[ $MODEL_PATH == *"gemma"* ]]; then
#     GPU_UTIL=0.65
# elif [[ $MODEL_PATH == *"Mistral"* ]]; then
#     GPU_UTIL=0.5
# else
#     GPU_UTIL=0.85
# fi
```
</details>

## Safety Evaluation

Safety evaluation requires running an evaluator LLM. Since vLLM currently does not natively support running multiple models in a single Python script, you need to first host the Llama-3.1-70B-Instruct model on your local server.

To host the 70B model (requires at least 4 GPUs such as A6000 48GB or A100 40/80GB), you can use the following command:

```bash
./scripts/eval/serve_llama.sh [GPU_IDS]
```

To run the safety evaluations, execute:

```bash
./scripts/eval/safety.sh [GPU_IDS] [MODEL_PATH]
```

## In-Context Response Learning

You can reproduce our in-context learning experiment results using the following command:

```bash
./scripts/eval/just_eval/just_eval_urial.sh [GPU_IDS] [HF_MODEL_PATH] [urial/urial_r/0shot]
```

## Acknowledgments

This work builds upon a multitude of existing studies and open-source projects. We would like to express our sincere gratitude to all the researchers and engineers who have contributed to and maintained these invaluable resources.

## Citation

If you find our results and code useful, please consider citing our paper:

```latex
@misc{an2024rt,
      title={Response Tuning: Aligning Large Language Models without Instruction}, 
      author={Seokhyun An and Hyounghun Kim},
      year={2024},
      eprint={2410.02465},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02465}, 
}
```