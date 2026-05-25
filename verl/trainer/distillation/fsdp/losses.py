# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}


def compute_forward_kl_topk_renorm(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward KL on top-k with renormalization over the top-k subset.

    Mathematically equivalent to ms-swift's GKDTrainer at beta=0 and matches
    the top-k variant in the GKD paper (Agarwal et al. 2023, Eq. 8). Both
    teacher and student distributions are renormalized to sum to 1 over the
    teacher's top-k indices before KL is computed:

        KL(P_tilde || Q_tilde) = sum_k P_tilde_k * (log P_tilde_k - log Q_tilde_k)

    where P_tilde_k = P_k / sum_{j in topk} P_j (same for Q_tilde).

    This differs from compute_forward_kl_topk above which uses full-vocab
    normalized probabilities restricted to top-k indices (P and Q do not sum
    to 1 over the subset). Empirically the renormalized form converges faster
    on sharp tasks because the per-token gradient magnitudes are not damped by
    the small full-vocab probabilities at the top-k indices.

    Diagnostic metrics (student_mass / teacher_mass) are still reported on the
    pre-renormalization full-vocab quantities, identical to forward_kl_topk,
    so dashboards stay comparable.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk). vLLM-provided full-vocab
            log-probabilities at the teacher's top-k indices.
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. full-vocab log-softmax + gather (same as forward_kl_topk; gives us the
    # full-vocab normalized log Q at topk indices, needed for student_mass diagnostic).
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs_full = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs_full.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    # 3. renormalize both distributions to sum to 1 over the topk subset.
    # log P_tilde_k = log P_full_k - logsumexp(log P_full over topk)
    # Equivalent to F.log_softmax on the gathered topk logits.
    student_log_probs_renorm = student_topk_log_probs_full - torch.logsumexp(
        student_topk_log_probs_full, dim=-1, keepdim=True
    )
    teacher_log_probs_renorm = teacher_topk_log_probs - torch.logsumexp(
        teacher_topk_log_probs, dim=-1, keepdim=True
    )

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_log_probs_renorm = student_log_probs_renorm.clamp_min(loss_config.log_prob_min_clamp)
        teacher_log_probs_renorm = teacher_log_probs_renorm.clamp_min(loss_config.log_prob_min_clamp)

    distillation_losses = kl_divergence(log_q=student_log_probs_renorm, log_p=teacher_log_probs_renorm)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}
