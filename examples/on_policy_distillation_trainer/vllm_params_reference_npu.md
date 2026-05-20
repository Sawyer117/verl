# Verl Sync OPD 可调 vllm 参数清单（NPU + FSDP + Qwen3.5 student/teacher）

> 适用脚本：`run_qwen3_8b_fsdp_npu.sh`（sync OPD 路径）
>
> 所有参数都通过 Hydra 命令行传给 `python -m verl.trainer.main_ppo`，分两组：
> - **学生 rollout** 走 `actor_rollout_ref.rollout.*`
> - **教师 inference** 走 `distillation.teacher_models.teacher_model.inference.*`
>
> 默认值参见 `verl/trainer/config/rollout/rollout.yaml` 和 `verl/trainer/config/distillation/distillation.yaml`。

## 0. 当前 sync OPD 跑通配置（基线）

```bash
# 学生 rollout
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.8
actor_rollout_ref.rollout.max_model_len=3073              # max_prompt(1024)+max_response(2048)+1
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.enforce_eager=false              # 默认走 ACL graph
actor_rollout_ref.rollout.enable_chunked_prefill=true
actor_rollout_ref.rollout.enable_prefix_caching=true
actor_rollout_ref.rollout.dtype=bfloat16
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288

# 教师 inference
distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8
distillation.teacher_models.teacher_model.inference.max_model_len=3073
distillation.teacher_models.teacher_model.inference.enforce_eager=true
distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=true
distillation.teacher_models.teacher_model.inference.enable_prefix_caching=true
```

---

## 1. 学生 Rollout VLLM 参数（`actor_rollout_ref.rollout.*`）

> 完整 schema：`verl/trainer/config/rollout/rollout.yaml`
> Python 配置类：`verl/workers/config/rollout.py` 的 `RolloutConfig`

### 1.1 并行度与资源

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `tensor_model_parallel_size` | `2` | vllm 内部 TP 切分。0.8B 学生 TP=1 即可；模型大可调到 2/4 | `rollout.yaml:56` |
| `data_parallel_size` | `1` | DP 副本数，可起多份 vllm engine 并发处理 prompt | `rollout.yaml:59` |
| `expert_parallel_size` | `1` | MoE EP 切分（only for MoE 学生）；ETP/EP 切换 | `rollout.yaml:62-64` |
| `pipeline_model_parallel_size` | `1` | vllm PP 切分。0.8B 用不到 | `rollout.yaml:67` |
| `nnodes` | `0` | 独立 rollout server 节点数。**sync 模式 = 0，async/one-step-off 才 > 0** | `rollout.yaml:11` |
| `n_gpus_per_node` | `${trainer.n_gpus_per_node}` | 每个 rollout 节点的 GPU 数 | `rollout.yaml:14` |

### 1.2 显存与 KV cache

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `gpu_memory_utilization` | `0.5` | vllm 在每张卡上拿走的显存比例（含权重 + KV cache）。你设 0.8 = 64GB×0.8 ≈ 51GB | `rollout.yaml:38` |
| `max_model_len` | `null` | 最大上下文长度（prompt + response）。`null` = 用模型默认；显式设 = 截 KV cache。**砍它能省 KV 显存换更大 batch** | `rollout.yaml:73` |
| `max_num_batched_tokens` | `8192` | 一个 forward batch 能调度的 token 总数（含 prefill）。chunked prefill 开后这是 chunk 上限 | `rollout.yaml:70` |
| `max_num_seqs` | `1024` | 同时 in-flight 的 sequence 上限。砍小防止 max_num_batched_tokens 不够分 | `rollout.yaml:76` |
| `free_cache_engine` | `True` | 训练 phase 时把 KV cache engine 销毁释放显存，wakeup 时重建 | `rollout.yaml:53` |

### 1.3 性能开关

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `enforce_eager` | `False` | 关掉 ACL/CUDA graph 用纯 eager。**调试用**，正常跑应该 False 让 graph 加速 | `rollout.yaml:44` |
| `enable_chunked_prefill` | `True` | 把长 prefill 切成块，跟 decode 交错调度，提高 throughput | `rollout.yaml:79` |
| `enable_prefix_caching` | `True` | 共享 prefix 的请求复用 KV cache，gsm8k 这种 prompt 短帮助有限但**没坏处保持开** | `rollout.yaml:82` |
| `cudagraph_capture_sizes` | `null` | 显式指定 CUDA graph capture 的 batch sizes（如 `[1,2,4,8,16,32]`）。**enforce_eager=False 时**有效，能省 capture 时显存 | `rollout.yaml:50` |
| `dtype` | `bfloat16` | 模型权重 dtype。bf16 是标准 | `rollout.yaml:35` |
| `disable_log_stats` | `True` | 关掉 vllm 自己的统计 log 输出。开就能看更多 vllm 内部 metric | `rollout.yaml:112` |
| `scheduling_policy` | `fcfs` | 调度策略：`fcfs`（先到先服务）/ `priority` | `rollout.yaml:88` |

### 1.4 采样参数（训练 rollout 用）

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `temperature` | `1.0` | 训练 rollout 采样温度。1.0 = 标准多样性 | `rollout.yaml:17` |
| `top_k` | `-1` | top-k 截断。`-1` = 关掉 | `rollout.yaml:20` |
| `top_p` | `1.0` | nucleus 采样。1.0 = 关掉 | `rollout.yaml:23` |
| `do_sample` | `True` | True 用采样，False 用 greedy | `rollout.yaml:116` |
| `ignore_eos` | `False` | 是否忽略 EOS 继续生成。一般 False | `rollout.yaml:41` |
| `n` | `1` | 每个 prompt 生成几条 response。GRPO 用 4/8，**OPD 通常 1** | `rollout.yaml:119` |
| `prompt_length` | `${data.max_prompt_length}` | 最大 prompt 长度，跟数据集对齐 | `rollout.yaml:27` |
| `response_length` | `${data.max_response_length}` | 最大 response 长度 | `rollout.yaml:31` |

### 1.5 采样参数（validation eval 用）

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `val_kwargs.temperature` | `0` | eval 采样温度 | `rollout.yaml:156` |
| `val_kwargs.top_k` | `-1` | eval top-k | `rollout.yaml:150` |
| `val_kwargs.top_p` | `1.0` | eval top-p | `rollout.yaml:153` |
| `val_kwargs.do_sample` | `False` | eval 是否采样（默认贪心） | `rollout.yaml:162` |
| `val_kwargs.n` | `1` | eval 时每 prompt 几条 | `rollout.yaml:159` |

### 1.6 weights 同步（FSDP → vllm）

> 完整 schema：`rollout.yaml:264-292`

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `checkpoint_engine.backend` | `naive` | 权重同步后端：`naive` / `nccl` / `nixl` / `hccl` | `rollout.yaml:270` |
| `checkpoint_engine.update_weights_bucket_megabytes` | `2048` | 每次 broadcast 的 bucket 大小（MB）。学生小可减；teacher 大可加。**目前 NPU 走 shm fallback** | `rollout.yaml:284` |
| `checkpoint_engine.engine_kwargs` | `{}` | 给 backend 传额外 kwargs | `rollout.yaml:287` |
| `checkpoint_engine.custom_backend_module` | `null` | 自定义 backend Python 路径 | `rollout.yaml:292` |

### 1.7 log_prob 重算（PPO 的 old_log_prob 阶段）

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `log_prob_use_dynamic_bsz` | 跟 actor 同 | 是否用 dynamic batching | `rollout.yaml:105` |
| `log_prob_max_token_len_per_gpu` | 跟 actor 同 | 一张卡上的 token 上限 | `rollout.yaml:109` |
| `log_prob_micro_batch_size` | `null` | 全局 micro batch size（**deprecated**） | `rollout.yaml:98` |
| `log_prob_micro_batch_size_per_gpu` | `null` | 单 GPU micro batch size | `rollout.yaml:101` |
| `calculate_log_probs` | `False` | 让 vllm 在生成时**顺手算 logprobs**（rollout correction bypass mode 用） | `rollout.yaml:225` |
| `logprobs_mode` | `processed_logprobs` | logprob 后处理方式 | `rollout.yaml:85` |

### 1.8 量化

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `quantization` | `null` | `fp8` / `torchao` / `null` | `rollout.yaml:447` |
| `quantization_config_file` | `null` | 量化 config 路径 | `rollout.yaml:450` |

### 1.9 Mode / 异步

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `mode` | `async` | `sync` / `async`。**sync OPD 用 `async`**（AgentLoop 模式） | `rollout.yaml:8` |
| `name` | `???` | `vllm` / `sglang` / `trtllm` / `hf` | `rollout.yaml:5` |
| `load_format` | `dummy` | `dummy` 随机初始化（之后由 FSDP sync 进真权重）/ `hf` / `safetensors` | `rollout.yaml:92` |
| `layered_summon` | `False` | FSDP 分层 unshard 省显存（同步慢） | `rollout.yaml:95` |
| `skip_tokenizer_init` | `True` | rollout 引擎跳过 tokenizer 初始化（token-in token-out） | `rollout.yaml:344` |
| `over_sample_rate` | `0` | 提前终止阈值：完成 (1 - rate) × N 就取消剩余 | `rollout.yaml:123` |
| `multi_stage_wake_up` | `False` | SGLang only（NPU OPD 用不上） | `rollout.yaml:128` |
| `enable_rollout_routing_replay` | `False` | MoE 路由 replay（0.8B 学生不是 MoE，关） | `rollout.yaml:348` |

### 1.10 AgentLoop（vllm server mode 调度层）

> sync OPD 在 verl 0.18+ 已经默认走 `mode=async` + AgentLoop 这套。

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `agent.num_workers` | `8` | 起几个 AgentLoop worker 并发处理 prompt | `rollout.yaml:234` |
| `agent.default_agent_loop` | `single_turn_agent` | 默认 agent 行为（非多轮工具调用） | `rollout.yaml:237` |
| `agent.agent_loop_config_path` | `null` | 自定义 agent loop 配置 | `rollout.yaml:249` |

### 1.11 vllm 引擎直传 kwargs（最灵活）

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.<任意 vllm 参数>=<值>
```

例如：

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=4    # CPU swap KV cache GB
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_async_output_proc=true
+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_lora=false
```

> 来源：`rollout.yaml:131-140`。任何 vllm CLI 支持但 verl 没显式列出的参数都可以这样塞进来。

---

## 2. 教师 Inference VLLM 参数（`distillation.teacher_models.teacher_model.inference.*`）

> 完整 schema：`verl/trainer/config/distillation/distillation.yaml:87-110`
>
> **教师跟学生用同一个 `RolloutConfig` 类**，所以学生侧 §1 的参数大部分都能在教师侧加上前缀使用。只是教师默认值不一样。

### 2.1 默认值跟学生不同的项

| Key | 教师默认 | 学生默认 | 来源 |
|---|---|---|---|
| `tensor_model_parallel_size` | `2` | `2` | `distillation.yaml:97` |
| `gpu_memory_utilization` | `0.5` | `0.5` | `distillation.yaml:91` |
| `enforce_eager` | **`true`** ← 教师默认强制 eager（避免 graph capture 开销） | `False` | `distillation.yaml:92` |
| `load_format` | `auto` ← 教师直接从 HF 加载 | `dummy` ← 学生先随机初始化再 sync | `distillation.yaml:101` |
| `max_num_seqs` | `1024` | `1024` | `distillation.yaml:100` |
| `prompt_length`, `response_length`, `temperature` | inherit from `actor_rollout_ref.rollout.*` | — | `distillation.yaml:108-110` |

### 2.2 教师独有

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `num_replicas` | `0` | 该教师起几份独立 vllm 副本。`0` 表示"不限定"，verl 自己算 | `distillation.yaml:86` |
| `limit_images` | `null` | 多模态 image 数限制（VL 模型用） | `distillation.yaml:103` |

### 2.3 学生有但教师未在默认列出的

教师默认 yaml **只列了一部分**字段（`distillation.yaml:88-110`），但因为 `_target_: verl.workers.config.RolloutConfig` 跟学生同类，**学生侧的字段都能在教师侧通过 `+` 强插入**：

```bash
+distillation.teacher_models.teacher_model.inference.scheduling_policy=priority
+distillation.teacher_models.teacher_model.inference.cudagraph_capture_sizes='[1,4,8]'
+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.swap_space=4
```

---

## 3. OPD 全局参数（教师资源池规模）

> 完整 schema：`verl/trainer/config/distillation/distillation.yaml:58-62`

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `distillation.n_gpus_per_node` | `8` | 教师资源池每节点 GPU 数 | `distillation.yaml:59` |
| `distillation.nnodes` | `0` | 教师资源池节点数 | `distillation.yaml:62` |
| `distillation.teacher_key` | `data_source` | 多教师路由 key | `distillation.yaml:113` |

---

## 4. OPD Distillation Loss 参数（**不是 vllm 但跟着用**）

> 完整 schema：`verl/trainer/config/distillation/distillation.yaml:17-55`

| Key | 默认 | 含义 | 来源 |
|---|---|---|---|
| `distillation.distillation_loss.loss_mode` | `k3` | `k1` / `k2` / `k3` / `forward_kl_topk` / `kl` / `abs` / `mse` / `low_var_kl` | `distillation.yaml:24` |
| `distillation.distillation_loss.topk` | `32` | 教师返回的 top-k 数 | `distillation.yaml:27` |
| `distillation.distillation_loss.use_task_rewards` | `True` | 是否把 task reward 加进 advantage | `distillation.yaml:30` |
| `distillation.distillation_loss.distillation_loss_coef` | `1.0` | use_task_rewards=True 时 KD loss 的权重 | `distillation.yaml:34` |
| `distillation.distillation_loss.loss_max_clamp` | `null` | KL 上下限 clamp（防 nan / 极端值） | `distillation.yaml:37` |
| `distillation.distillation_loss.log_prob_min_clamp` | `null` | log prob 下限 clamp | `distillation.yaml:40` |
| `distillation.distillation_loss.use_policy_gradient` | `False` | 是否走 PPO 框架（PG-style）。False = GKD-style 直接 backprop | `distillation.yaml:43` |
| `distillation.distillation_loss.policy_loss_mode` | `vanilla` | PG-style 用哪种 policy loss（目前只支持 vanilla） | `distillation.yaml:46` |
| `distillation.distillation_loss.clip_ratio` | `0.2` | PG-style 的 PPO clip | `distillation.yaml:49` |
| `distillation.distillation_loss.clip_ratio_low` | `0.2` | PPO clip 下界 | `distillation.yaml:52` |
| `distillation.distillation_loss.clip_ratio_high` | `0.2` | PPO clip 上界 | `distillation.yaml:55` |

---

## 5. NPU 上特别值得调的几个（行动建议）

| 优先级 | Key | 当前 | 建议 | 理由 |
|---|---|---|---|---|
| 🟢 High | `actor_rollout_ref.rollout.cudagraph_capture_sizes` | `null`（自动） | `[1,2,4,8,16,32,64]` | 减少 graph capture 显存，提高启动稳定性 |
| 🟢 High | `distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size` | 2 | **4** | 单份 teacher 4 卡 vs 2 份 × 2 卡：减少教师空载，加快 logprob 取回 |
| 🟡 Mid | `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.8 | 试 0.85 | 学生 0.8B 权重很小，多腾出来给 KV → 更大 batch |
| 🟡 Mid | `actor_rollout_ref.rollout.max_num_batched_tokens` | 8192 | 试 16384 | chunked prefill 块更大，rollout 长尾稍微好一点 |
| 🟡 Mid | `distillation.distillation_loss.topk` | 64 | 试 32 | 实测 teacher_mass 已经 99.6%，top-32 大概率也够，省传输 |
| 🟢 High | `distillation.teacher_models.teacher_model.inference.enforce_eager` | `True`（教师默认） | 保持 `True` | 教师不走 ACL graph 启动更快、内存更省，吞吐影响小 |
| 🔵 Low | `actor_rollout_ref.rollout.enable_rollout_routing_replay` | `False` | 保持 | 0.8B 学生非 MoE，无关 |
| 🔵 Low | `actor_rollout_ref.rollout.scheduling_policy` | `fcfs` | 保持 | 公平调度，gsm8k batch 内长尾不严重 |

---

## 6. 命令行模板（综合所有可调项）

```bash
HYDRA_FULL_ERROR=1 \
WANDB_MODE=offline \
... bash run_qwen3_8b_fsdp_npu.sh \
  \
  `# === 学生 rollout 并行 / 资源 ===` \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.max_model_len=3073 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  \
  `# === 学生 rollout 性能 ===` \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enable_prefix_caching=true \
  actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,32]' \
  \
  `# === 学生 rollout 采样 ===` \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n=1 \
  \
  `# === 学生 rollout 权重同步 ===` \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  \
  `# === 学生 rollout 直传 vllm 任意 kwarg ===` \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=4 \
  \
  `# === 教师 inference 并行 / 资源 ===` \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=4 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
  distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=16384 \
  distillation.teacher_models.teacher_model.inference.enforce_eager=true \
  distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=true \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=true \
  \
  `# === 教师资源池 ===` \
  distillation.n_gpus_per_node=4 \
  distillation.nnodes=1 \
  \
  `# === OPD loss 配置 ===` \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_policy_gradient=true \
  distillation.distillation_loss.use_task_rewards=false \
  distillation.distillation_loss.clip_ratio=0.2
```

---

## 7. 出处参考

| 路径 | 内容 |
|---|---|
| `verl/trainer/config/rollout/rollout.yaml` | RolloutConfig 完整 schema + 注释 |
| `verl/workers/config/rollout.py` | RolloutConfig 的 Python dataclass 定义 |
| `verl/trainer/config/distillation/distillation.yaml` | DistillationConfig 完整 schema |
| `verl/workers/config/distillation.py` | DistillationConfig 的 Python dataclass + post_init 检查（k1+GKD 硬卡那段） |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | vllm engine 实际启动逻辑 |
| `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | FSDP → vllm 权重同步细节 |
| `verl/experimental/teacher_loop/teacher_manager.py` | 教师 vllm worker 管理逻辑 |
