DEVICE=$1
MODEL_PATH=$2

NUM_GPUS=$(echo $DEVICE | tr ',' '\n' | wc -l)

echo "CUDA_DEVICE_IDX: $DEVICE"
echo "NUM_GPUS: $NUM_GPUS"

if [[ "$MODEL_PATH" == *"outputs"* ]]; then
    SAVE_DIR_ROOT="$MODEL_PATH/evals"
else
    SAVE_DIR_ROOT="outputs/remote_models/$(basename "$MODEL_PATH")/evals"
    mkdir -p "$SAVE_DIR"
fi

if [[ $MODEL_PATH == *"gemma"* ]]; then
    export VLLM_ATTENTION_BACKEND="FLASHINFER"
fi

GPU_UTIL=0.85
BATCH_SIZE=auto

# Uncomment the lines below if you encounter an CUDA OOM error.
# if [[ $MODEL_PATH == *"gemma"* ]]; then
#     GPU_UTIL=0.65
#     BATCH_SIZE=1
# elif [[ $MODEL_PATH == *"Mistral"* ]]; then
#     GPU_UTIL=0.5
#     BATCH_SIZE=1
# else
#     GPU_UTIL=0.85
#     BATCH_SIZE=auto
# fi

echo "RUN GSM8K EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/gsm8k_mt8shot"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks gsm8k_cot_llama \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 8 \
    --fewshot_as_multiturn \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"

echo "RUN AI2_ARC EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/ai2_arc"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks ai2_arc \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 0 \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"

echo "RUN HELLASWAG EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/hellaswag"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks hellaswag \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 0 \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"

echo "RUN openbookqa EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/openbookqa"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks openbookqa \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 0 \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"

echo "RUN piqa EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/piqa"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks piqa \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 0 \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"

echo "RUN mmlu EVAL"
SAVE_DIR="$SAVE_DIR_ROOT/mmlu"
CUDA_VISIBLE_DEVICES=$DEVICE lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${GPU_UTIL} \
    --tasks mmlu \
    --apply_chat_template \
    --log_samples \
    --output_path $SAVE_DIR \
    --num_fewshot 0 \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "temperature=0"