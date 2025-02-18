# Revealing the Inherent Instructability of Pre-Trained Language Models

This repository contains the code and resources supporting the paper "[Revealing the Inherent Instructability of Pre-Trained Language Models](https://arxiv.org/abs/2410.02465)" by [Seokhyun An](https://seokhyunan.com), Minji Kim and [Hyounghun Kim](https://hyounghk.github.io/).


## Overview

We propose Response Tuning (RT) to verify our hypothesis that the ability to process instructions can be developed in the pre-training stage. Unlike instruction tuning, RT does not condition the response tokens on the paired instruction, which precludes the model from learning to generate responses according to instructions. Rather, it focuses on learning the response distribution.

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

To reproduce our refusal experiments, create an IT dataset mixed with refusal examples, and train the model using this dataset:

```bash
python3 generate_refusal_mixture.py \
    --target_dataset [TRAINING_DATASET_PATH] \
    --num_safety_examples [NUMBER_OF_SAFETY_EXAMPLES_TO_MIX]
```

<details>
<summary>Example</summary>

```bash
python3 generate_refusal_mixture.py \
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

## Refusal Evaluation

Refusal evaluation requires running an evaluator LLM. Since vLLM currently does not natively support running multiple models in a single Python script, you need to first host the Llama-3.1-70B-Instruct model on your local server.

To host the 70B model (requires at least 4 GPUs such as A6000 48GB or A100 40/80GB), you can use the following command:

```bash
./scripts/eval/serve_llama.sh [GPU_IDS]
```

To run the refusal evaluations, execute:

```bash
./scripts/eval/refusal.sh [GPU_IDS] [MODEL_PATH]
```

## In-Context Response Learning

You can reproduce our in-context learning experiment results using the following command:

```bash
./scripts/eval/just_eval/just_eval_urial.sh [GPU_IDS] [HF_MODEL_PATH] [urial/urial_r/0shot]
```

## Citation

If you find our results and code useful, please consider citing our paper:

```latex
@misc{an2025revealing,
      title={Revealing the Inherent Instructability of Pre-Trained Language Models}, 
      author={Seokhyun An and Minji Kim and Hyounghun Kim},
      year={2025},
      eprint={2410.02465},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02465}, 
}
```