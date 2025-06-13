#!/bin/bash
set -x

ulimit -n 1048576

export MODEL_PATH="/home/ducati/projects/finalproject/rllm/checkpoints/deepcoder/1.5b-base-cot-nodoc/actor/global_step_8"
export WANDB_API_KEY="c2028e9b42ea9ced013b78fe33aec2a01fbf4a82"
export WANDB_ENTITY="ducati-tsinghua-university"

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

export RAY_worker_niceness=0
export RAY_memory_monitor_refresh_ms=0
export TOKENIZERS_PARALLELISM=false

pkill -f "ray::"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/root/autodl-tmp/data/ours_train_nodoc.json \
    data.val_files=./data/test_livecodebench.json \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.compute_reward=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepcoder' \
    trainer.experiment_name='1.5b-base-cot-nodoc' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=8 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}" 

# Train over 1 node with 2 GPUs.
# - Modifications for testing purposes:
#   - trainer.nnodes=1 (originally 4)
#   - trainer.n_gpus_per_node=2 (originally 8)
#   - data.max_response_length=4096 (originally 16384)
#   - data.train_batch_size=1 (originally 32)
#   - actor_rollout_ref.rollout.n=1 (originally 8)
#   - actor_rollout_ref.rollout.gpu_memory_utilization=0.5 (originally 0.8)
#   - actor_rollout_ref.actor.ppo_mini_batch_size=1 (originally 16)
#   - actor_rollout_ref.actor.ppo_micro_batch_size=1 (originally 16)
#   - trainer.total_epochs=1 (originally 100)
# - Other modifications:
#   - data.max_prompt_length=4096 (originally 2048)
#   - actor_rollout_ref.rollout.compute_reward=False (originally True)
#     - This is due to a bug that causes the model to always use NaiveRewardManager(verl/verl/workers/reward_manager/naive.py) rather than the custom reward manager(verl/verl/trainer/main_ppo.py) when actor_rollout_ref.rollout.compute_reward=True(verl/verl/trainer/ppo/ray_trainer.py:L622, verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py:L287).
