DEVICES=$1
BASE_MODEL=$2
DATASET_PATH=$3

CUDA_VISIBLE_DEVICES=$DEVICES python3 hf_ft.py \
    --model $BASE_MODEL \
    --dataset_path $DATASET_PATH \
    --tuning_type it