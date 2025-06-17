#!/bin/bash
set -x

ulimit -c 0

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# This experiment was conducted with vLLM version 0.8.2.
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export VLLM_USE_V1=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/path/to/16k_neg"
fi

MODEL_NAME=24k_neg
mkdir -p deepscaler/checkpoints/deepscaler/$MODEL_NAME
cp -r scripts/train/24k_neg.sh deepscaler/checkpoints/deepscaler/$MODEL_NAME
cd deepscaler

# We mask out all positive samples and update the policy model using only negative samples by setting cut_down_positive_samples=True.
# We set cut_down_samples_reserve_ratio=0.0 and cut_down_samples_reserve_min_num=0.0 to ensure that no positive examples are retained.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.cut_down_positive_samples=True \
    algorithm.cut_down_samples_reserve_ratio=0.0 \
    algorithm.cut_down_samples_reserve_min_num=0.0 \
    data.train_files=/absolute/path/to/data/train/deepscaler-train.dedup_long.parquet \
    data.val_files=/absolute/path/to/data/train/aime.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=24576 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.n_val=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name=$MODEL_NAME \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"