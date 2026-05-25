#!/usr/bin/env bash
# On-policy distillation | Ascend NPU | FSDP backend | renorm vs full-vocab forward KL ablation
#
# Purpose: A/B compare the two GKD-style supervised distillation losses on the
# *same* config to isolate whether renormalization-over-topk (ms-swift / GKD
# paper Eq. 8 convention) explains the convergence gap observed against
# ms-swift on identical hyperparameters.
#
# Pinned (vs run_qwen3_8b_fsdp_npu.sh):
#   USE_POLICY_GRADIENT=False                   # supervised GKD, not PG
#   distillation_loss_mode in:
#     - forward_kl_topk          (verl-native, softmax-then-gather, full-vocab norm)
#     - forward_kl_topk_renorm   (new, gather-then-softmax, renormed over topk)
#
# Two runs back-to-back, separate experiment_name so logs don't collide.
#
# Usage:
#   bash run_qwen3_8b_fsdp_npu_renorm_ablation.sh                # runs both
#   LOSS_MODES=forward_kl_topk_renorm bash run_qwen3_8b_fsdp_npu_renorm_ablation.sh   # just renorm
#
# Override anything via env, see run_qwen3_8b_fsdp_npu.sh for the full list.

set -xeuo pipefail

export HYDRA_FULL_ERROR=1

# ---- defaults aligned with the colleague's ms-swift recipe ----
STUDENT_MODEL=${STUDENT_MODEL:-/models/Qwen2.5-0.5B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-/models/Qwen3-8B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

# Supervised GKD pinned, matches ms-swift --rlhf_type gkd --lmbda 1 --beta 0.
use_policy_gradient=False
distillation_topk=${DISTILLATION_TOPK:-64}

# Loss modes to run sequentially. Set LOSS_MODES env to run only one.
LOSS_MODES=${LOSS_MODES:-"forward_kl_topk forward_kl_topk_renorm"}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}
max_steps=${MAX_STEPS:-100}

project_name=${PROJECT_NAME:-verl_distill_renorm_ablation}

GSM8K_DIR=${GSM8K_DIR:-/workspace/verl_data/gsm8k}
gsm8k_train=${GSM8K_TRAIN:-${GSM8K_DIR}/train.parquet}
gsm8k_test=${GSM8K_TEST:-${GSM8K_DIR}/test.parquet}

train_files="['$gsm8k_train']"
val_files="['$gsm8k_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays (shared across runs) ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER_COMMON=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${max_steps}
)

DISTILL_COMMON=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch each loss mode ###########################
for loss_mode in ${LOSS_MODES}; do
    experiment_name="qwen3_8b_fsdp_npu_${loss_mode}"
    echo "================================================================"
    echo "Running loss_mode=${loss_mode}  experiment_name=${experiment_name}"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        "${DATA[@]}" \
        "${MODEL[@]}" \
        "${ACTOR[@]}" \
        "${ROLLOUT[@]}" \
        "${TRAINER_COMMON[@]}" \
        trainer.experiment_name=${experiment_name} \
        "${DISTILL_COMMON[@]}" \
        distillation.distillation_loss.loss_mode=${loss_mode} \
        "$@"
done
