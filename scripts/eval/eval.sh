# set -e
cd scripts/eval
export VLLM_USE_V1=0

MODEL=/XXXXX/model_name/global_step_XXXX

###################
DATA=aime
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA
bash vllm_general_reasoning_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=amc
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA
bash vllm_general_reasoning_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=aime
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-8k
bash vllm_8k_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=amc
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-8k
bash vllm_8k_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=gpqa_diamond
REPEAT=32
OUTPUT_DIR=$MODEL/$DATA
bash vllm_multiple_choice_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=mmlu_stem
REPEAT=4
OUTPUT_DIR=$MODEL/$DATA
bash vllm_multiple_choice_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 220
wait

###################
DATA=aime25
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA
bash vllm_general_reasoning_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200
wait

###################
DATA=gaokao_math_qa
REPEAT=32
OUTPUT_DIR=$MODEL/$DATA
bash vllm_multiple_choice_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 220
wait

###################
DATA=amc
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-MS1_100
bash vllm_robustness_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200 1
wait

###################
DATA=aime
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-MS1_100
bash vllm_robustness_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200 1
wait

###################
DATA=amc
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-MS2_100
bash vllm_robustness_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200 2
wait

###################
DATA=aime
REPEAT=64
OUTPUT_DIR=$MODEL/$DATA-MS2_100
bash vllm_robustness_test.sh $MODEL/actor/huggingface $OUTPUT_DIR $DATA $REPEAT 200 2
wait
