MODEL_PATH=$1

REG_OUTPUT_PATH="$1/evals/just_eval/just_eval_regular_eval_output.json"
SAFETY_OUTPUT_PATH="$1/evals/just_eval/just_eval_safety_eval_output.json"

if [ ! -f $REG_OUTPUT_PATH ]; then
    echo "Regular evaluation output not found at $REG_OUTPUT_PATH"
    exit 1
fi

# Check if safety evaluation output exists
if [ ! -f $SAFETY_OUTPUT_PATH ]; then
    python3 -m eval.just_eval.compute_metrics --regular_eval_output_path $REG_OUTPUT_PATH
else
    python3 -m eval.just_eval.compute_metrics --regular_eval_output_path $REG_OUTPUT_PATH --safety_eval_output_path $SAFETY_OUTPUT_PATH
fi