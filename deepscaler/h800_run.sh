cd /mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/verl
pip install -i https://mirrors.tencent.com/pypi/simple/ -e .

cd /mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler
pip install -i https://mirrors.tencent.com/pypi/simple/ -e .

cd /mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k-2/actor/global_step_560"
nohup sh ./scripts/train/run_deepscaler_1.5b_16k-myy2.sh --model $MODEL_PATH > /mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/outputs/1.5b-16k-reserve0.05_512_after_positive.log 2>&1 &


wait


trap 'wait' SIGCHLD && sleep 365d && wait

# bash /mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/h800_run.sh