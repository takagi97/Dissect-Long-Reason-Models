# set -e

cd scripts/eval
export VLLM_USE_V1=0

MODEL=$1
OUTPUT_DIR=$2
DATA=$3
REPEAT=$4
CONCURRENCY=$5
PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1 2 3 4 5 6 7)

if [[ -d "$OUTPUT_DIR" ]]; then
    # mv $OUTPUT_DIR $OUTPUT_DIR.old
    echo "Output directory $OUTPUT_DIR already exists. Exiting..."
    exit 1
fi

for i in "${!DEVICES[@]}"; do
    PORT="${PORTS[$i]}"
    DEVICE="${DEVICES[$i]}"

    CUDA_VISIBLE_DEVICES=$DEVICE vllm serve $MODEL \
    --max_model_len 32768 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --port $PORT &
done

sleep 2m

mkdir -p $OUTPUT_DIR
cp vllm_16k_test.sh vllm_16k_test.py $OUTPUT_DIR
python vllm_16k_test.py \
    --model $MODEL \
    --file /absolute/path/to/data/test/$DATA.json \
    --ports 8010,8011,8012,8013,8014,8015,8016,8017 \
    --repeat $REPEAT \
    --concurrency $CONCURRENCY \
    --output_dir $OUTPUT_DIR
