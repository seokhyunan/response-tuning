DEVICE=$1
DATASET_PATH=$2

NUM_GPUS=$(echo $DEVICE | tr ',' '\n' | wc -l)

echo "CUDA_DEVICE_IDX: $DEVICE"
echo "NUM_GPUS: $NUM_GPUS"

python3 refine_responses.py \
    --dataset_path $DATASET_PATH \
    --refiner "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --refiner_engine_backend "vllm" \
    --refiner_num_gpus $NUM_GPUS