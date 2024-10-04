DEVICE=$1
MODEL_PATH=$2

NUM_GPUS=$(echo $DEVICE | tr ',' '\n' | wc -l)

echo "CUDA_DEVICE_IDX: $DEVICE"
echo "NUM_GPUS: $NUM_GPUS"

# if "gemma" in the model path, use FLASHINFER, else use VLLM
if [[ $MODEL_PATH == *"gemma"* ]]; then
    export VLLM_ATTENTION_BACKEND="FLASHINFER"
fi

GPU_UTIL=0.85
# Uncomment the lines below if you encounter an CUDA OOM error.
# if [[ $MODEL_PATH == *"gemma"* ]]; then
#     GPU_UTIL=0.65
# elif [[ $MODEL_PATH == *"Mistral"* ]]; then
#     GPU_UTIL=0.5
# else
#     GPU_UTIL=0.85
# fi

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval.just_eval.run_inference \
    --model $MODEL_PATH \
    --use_chat_format \
    --model_num_gpus $NUM_GPUS \
    --gpu_memory_utilization $GPU_UTIL