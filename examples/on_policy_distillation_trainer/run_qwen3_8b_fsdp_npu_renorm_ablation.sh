#!/usr/bin/env bash
# On-policy distillation | Ascend NPU | FSDP backend | renorm vs full-vocab forward KL ablation
#
# Purpose: A/B compare the two GKD-style supervised distillation losses on the
# *same* config to isolate whether renormalization-over-topk (ms-swift / GKD
# paper Eq. 8 convention) explains the convergence gap observed against
# ms-swift on identical hyperparameters.
#
# Defaults below are pinned to the user's NPU box and aligned with the
# ms-swift GKD recipe (table reference: VERL_MS-SWIFT_GKD_参数配置.xlsx):
#   - Qwen3.5-0.8B (student) <- Qwen3.5-35B-A3B (teacher)
#   - 12 student NPU + 4 teacher NPU (TP=4) on a single node
#   - prompt 4096 + response 2048, lr 1e-5, warmup 5%, max_steps 100
#   - β=0 forward KL, λ=1 fully on-policy, top-k=64
#   - GSM8k preprocessed to /workspace/verl_data/gsm8k/{train,test}.parquet
#
# Pinned for the ablation (cannot be overridden cleanly via env):
#   USE_POLICY_GRADIENT=False                   # supervised GKD, not PG
#   distillation_loss_mode in:
#     - forward_kl_topk          (verl-native, softmax-then-gather, full-vocab norm)
#     - forward_kl_topk_renorm   (new, gather-then-softmax, renormed over topk)
#
# Two runs back-to-back, separate experiment_name so logs don't collide.
#
# Usage (the way the user has been running their smoke):
#   bash run_qwen3_8b_fsdp_npu_renorm_ablation.sh                # both modes, 100 steps each
#   LOSS_MODES=forward_kl_topk_renorm \
#     MAX_STEPS=2 bash run_qwen3_8b_fsdp_npu_renorm_ablation.sh  # smoke renorm only, 2 steps
#
# Override anything via env, see the block below for the full list.

set -xeuo pipefail

# ---- NPU runtime env (mirrors the user's smoke setup) ----
export HYDRA_FULL_ERROR=1
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-ASCEND}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export WANDB_MODE=${WANDB_MODE:-offline}

# ---- model + topology (matches user's smoke + ms-swift) ----
STUDENT_MODEL=${STUDENT_MODEL:-/home/canada_group_account/a00652497/model/qwen3.5_0.8B}
TEACHER_MODEL=${TEACHER_MODEL:-/home/canada_group_account/a00652497/model/qwen3.5_35B_a3B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-12}        # ms-swift: 12 student NPU
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4} # ms-swift: 4 teacher NPU
TEACHER_TP=${TEACHER_TP:-4}                 # ms-swift: TP=4 for MoE
ROLLOUT_TP=${ROLLOUT_TP:-1}                 # student 0.8B fits on one rank

# ---- loss config (pinned for supervised GKD ablation) ----
use_policy_gradient=False
distillation_topk=${DISTILLATION_TOPK:-64}                       # ms-swift gkd_logits_topk=64
LOSS_MODES=${LOSS_MODES:-"forward_kl_topk forward_kl_topk_renorm"}

# ---- batch sizing (aligned to ms-swift: per_device=4, GRAD_ACC=1, 12 GPUs -> effective 48) ----
train_batch_size=${TRAIN_BATCH_SIZE:-48}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-48}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}                     # ms-swift max_length=4096
max_response_length=${MAX_RESPONSE_LENGTH:-2048}                 # ms-swift max_completion_length=2048
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}    # matches the user's smoke

# ---- optim (ms-swift: lr=1e-5, warmup 5%) ----
actor_lr=${ACTOR_LR:-1e-5}
warmup_ratio=${WARMUP_RATIO:-0.05}

# ---- vllm gpu mem (ms-swift: student 0.65, teacher 0.85; user's smoke had 0.8/0.8) ----
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}

# ---- training horizon (ms-swift: 100 steps, no eval/save mid-run) ----
total_epochs=${TOTAL_EPOCHS:-15}        # large enough so MAX_STEPS dominates
max_steps=${MAX_STEPS:-100}
save_freq=${SAVE_FREQ:--1}              # disabled during ablation
test_freq=${TEST_FREQ:--1}              # disabled during ablation; eval at the end if needed

# ---- temperature, smoke-style stability switches (matches user's prior smoke) ----
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}                  # ms-swift temperature=1.0
use_torch_compile=${USE_TORCH_COMPILE:-False}                    # smoke off; flip to True once stable

# ---- logging ----
project_name=${PROJECT_NAME:-verl_distill_renorm_ablation}
LOG_DIR=${LOG_DIR:-/home/canada_group_account/a00652497/bytedance/post-train/logs}
mkdir -p "${LOG_DIR}"

# ---- data ----
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
    data.truncation='right'                    # match ms-swift --truncation_strategy right
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=${use_torch_compile}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${warmup_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enforce_eager=True              # ms-swift --vllm_enforce_eager
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
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
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
    experiment_name="qwen3.5_0.8b_from_35b_a3b_${loss_mode}"
    log_file="${LOG_DIR}/verl_${loss_mode}_$(date +%Y%m%d_%H%M%S).log"
    echo "================================================================"
    echo "  loss_mode        = ${loss_mode}"
    echo "  experiment_name  = ${experiment_name}"
    echo "  log_file         = ${log_file}"
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
        "$@" \
        2>&1 | tee "${log_file}"

    echo "log saved to: ${log_file}"
done
