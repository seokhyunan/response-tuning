DEVICE=$1
MODEL_PATH=$2
RPATH=$3

NUM_GPUS=$(echo $DEVICE | tr ',' '\n' | wc -l)

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

ARGS="--model $MODEL_PATH --model_num_gpus $NUM_GPUS --use_chat_format --gpu_memory_utilization $GPU_UTIL"

if [[ "$RPATH" != "" ]]; then
    ARGS="$ARGS --reference_path $RPATH"
fi

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval.alpaca_eval.run_eval $ARGS