# Relax 项目记忆文档

## 目标任务

对 `Qwen3.5-122B-A10B` 做**多模态 RL 训练**。

模型路径：`/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B`

GPU 资源：最多千卡可用，典型配置 128 卡（2节点×64 或 16节点×8），最小配置 32 卡（单机 4节点×8 或单机 32）。

---

## 目标模型：Qwen3.5-122B-A10B

### 架构信息（来自 config.json）

| 字段 | 值 |
|------|----|
| architectures | `Qwen3_5MoeForConditionalGeneration` |
| model_type | `qwen3_5_moe` |
| 语言主干层数 | 48 层（text_config.num_hidden_layers） |
| hidden_size | 3072 |
| num_experts | 256，num_experts_per_tok=8 |
| 注意力头 | 32 heads，num_key_value_heads=2，head_dim=256 |
| 层类型 | **混合架构**：linear_attention + full_attention 交替（每4层一个 full_attention）|
| vocab_size | 248320 |
| max_position_embeddings | 262144 |
| MTP 层 | mtp_num_hidden_layers=1 |

**这是 VLM（多模态模型），不是纯文本模型。**

### Vision Encoder（vision_config）

| 字段 | 值 |
|------|----|
| depth | 27 层 |
| hidden_size | 1152 |
| num_heads | 16 |
| patch_size | 16 |
| in_channels | 3 |
| spatial_merge_size | 2 |
| temporal_patch_size | 2 |
| out_hidden_size | 3072（与语言模型 hidden_size 对齐）|
| num_position_embeddings | 2304 |

### 特殊 Token

| Token | ID |
|-------|----|
| image_token_id | 248056 |
| vision_start_token_id | 248053 |
| vision_end_token_id | 248054 |
| video_token_id | 248057 |

### 位置编码

M-RoPE（多轴 RoPE），mrope_section=[11, 11, 10]，mrope_interleaved=true，rope_theta=10000000，partial_rotary_factor=0.25。

**训练必须加 `--no-rope-fusion`**（M-RoPE 返回 tensor list，与 fused kernel 不兼容，否则导致训练-推理 log-prob 不匹配）。

---

## 多模态 vs 纯文本本质区别

### 架构差异

```
纯文本: token_ids → Embedding → Transformer → LM Head

多模态: token_ids ──────────────────────────────┐
                                                 ▼
        pixel_values → Vision Encoder → embeds → 替换 image_token → Transformer → LM Head
```

VLM 额外组件：Vision Encoder、视觉投影层、图像特殊 token、M-RoPE 位置编码。

### 训练数据流差异

**Rollout 阶段**（性能开销最大）：
- 纯文本：tokenize → payload → SGLang
- 多模态：tokenize + processor(pixel_values) + base64编码 → payload + image_data → SGLang

**训练阶段**（forward 额外输入）：
- 纯文本：`(tokens, position_ids, attention_mask)` → loss
- 多模态：`(tokens, position_ids, attention_mask, pixel_values, image_grid_thw)` → vision_encode → 注入 → loss
- `image_grid_thw`：每张图 [T, H, W] 网格维度，用于 M-RoPE 3D 位置编码
- Vision encoder 参数通常冻结，不参与 RL loss 反传

### M-RoPE 位置编码差异

```
纯文本 RoPE:    position_ids = [0, 1, 2, 3, ...]          # 1D tensor

多模态 M-RoPE:  position_ids = [
    [t_0, t_1, ..., t_n],   # 时间轴（11 head dims）
    [h_0, h_1, ..., h_n],   # 高度轴（11 head dims）
    [w_0, w_1, ..., w_n],   # 宽度轴（10 head dims）
]                                                           # Python list of tensors
```

文本 token 三轴相同（退化为 1D），图像 token 三轴由 image_grid_thw 解析。

---

## Relax 框架对 Qwen3.5 VLM 的支持情况

### 支持状态：完整原生支持

| 组件 | 状态 | 路径 |
|------|------|------|
| 权重转换 | ✅ | `relax/backends/megatron/weight_conversion/qwen3_5.py` |
| GDN linear attention 转换 | ✅ | 同上（gdn_hf_to_mca / gdn_mca_to_hf） |
| Bridge 模式 | ✅ | `relax/backends/megatron/model_provider.py` |
| 多模态配置 | ✅ | `relax/utils/multimodal/config.py` |
| 多模态参数 | ✅ | `relax/utils/arguments.py` 第739-817行 |
| 多模态处理 | ✅ | `relax/utils/multimodal/`（image/video/audio utils）|
| 现成训练脚本 | ✅ | `scripts/training/multimodal/run-qwen35-*.sh` |

### 已有参考脚本

| 脚本 | 规模 | 模式 | 数据 |
|------|------|------|------|
| `run-qwen35-35B-A3B-8xgpu.sh` | 8xGPU | colocate | openr1mm（图像）|
| `run-qwen35-9B-8xgpu-openr1mm-async.sh` | 8xGPU | async | openr1mm（图像）|
| `run-qwen35-9B-8xgpu-video.sh` | 8xGPU | - | 视频 |

最接近目标的参考脚本：**`run-qwen35-35B-A3B-8xgpu.sh`**（同为 MoE 架构，含多模态配置）。

### 已有文本基线脚本（规模更大，可参考并行配置）

`scripts/training/text/run-qwen35-35B-A3B-16xgpu-async.sh`
- TP=2, PP=1, EP=8，16xGPU，fully async

### 关键多模态参数

```bash
--multimodal-keys '{"image":"image"}'   # 必填，声明数据字段映射
--no-rope-fusion                         # 必填，M-RoPE 兼容性
--megatron-to-hf-mode bridge             # 必填，启用 bridge 模式
--image-max-token-num 16384              # 可选，图像最大 token 数
--image-min-token-num 4                  # 可选，图像最小 token 数
--mm-processor-pool-size N               # 可选，多模态处理器池
--freeze-vision-model                    # 可选，冻结视觉编码器
```

### 已有 qwen35-35B-A3B 模型配置

`scripts/models/qwen35-35B-A3B.sh`：MoE 参数（TP/PP/EP、num-experts=256、moe-ffn-hidden-size 等）。

---

## 122B-A10B Megatron 模型配置（已验证）

文件：`scripts/models/qwen35-122B-A10B.sh`

与 35B-A3B 的差异：

| 参数 | 35B-A3B | 122B-A10B |
|------|---------|-----------|
| `--num-layers` | 40 | 48 |
| `--hidden-size` | 2048 | 3072 |
| `--num-attention-heads` | 16 | 32 |
| `--ffn-hidden-size` | 512 | 1024 |
| `--moe-ffn-hidden-size` | 512 | 1024 |
| `--moe-shared-expert-intermediate-size` | 512 | 1024 |

相同：`--num-experts 256`、`--moe-router-topk 8`、`--num-query-groups 2`、`--kv-channels 256`、`--vocab-size 248320`、`--rotary-percent 0.25`、`--rotary-base 10000000`、`--attention-output-gate`、`--moe-shared-expert-gate`

**attn_output_gate=True 的影响**：Q projection 实际输出维度为 `num_query_groups * value_per_group * 2 * head_dim = 16384`（不是 `num_heads * head_dim = 8192`），权重转换代码已正确处理。

## 权重转换验证（已通过）

测试文件：`tests/backends/megatron/test_qwen35_122B_weight_conversion.py`

**71/71 测试全部通过**，覆盖：

| 测试类 | 覆盖内容 |
|--------|---------|
| `TestGlobalParams` | embed_tokens、lm_head、final_layernorm |
| `TestFullAttentionLayers` | QKV 分离（维度/numel）、o_proj、QK norm、input_layernorm |
| `TestLinearAttentionLayers` | GDN roundtrip、in_proj→4 HF keys、out_norm +1 offset、A_log/dt_bias 透传 |
| `TestMoELayers` | grouped/ungrouped experts、shared expert、router weight/bias |
| `TestLayerIndexCoverage` | 跨全部 48 层的路由验证 |
| `TestErrorHandling` | 未知参数名报 ValueError |

**关键发现**：

1. **transformers 版本**：当前环境 4.57.1 不支持 `qwen3_5_moe`，需升级到 5.3.0（requirements.txt 已要求）。`get_hf_config` 在加载 GDN in_proj 时会触发此限制，生产环境需确保 transformers>=5.3.0。

2. **GDN 线性注意力**：122B-A10B 有 36 个 linear_attention 层，GDN 参数尺寸：
   - qk_dim = 128 × 16 = 2048
   - v_dim = 128 × 64 = 8192
   - MCA tensor shape: `(qk_dim*2 + v_dim*2 + num_v_heads*2, hidden)` = `(20544+128, 3072)`

3. **full_attention 层索引（0-based）**：3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47（共 12 层）

---

## 框架能力评估（技术尽调）

基于对 Relax 代码库的逐文件分析，记录各能力项的实际支持状态与关键约束。

### 模块一：模型适配与底层性能基建

#### 1. EP（专家并行）支持

**训练侧（Megatron）：原生支持**
- `relax/backends/megatron/initialize.py:46-48`：显式初始化 `expert_model_parallel_group` 和 `expert_tensor_parallel_group`
- `relax/backends/megatron/arguments.py:38-44`：强制 `variable_seq_lengths=True`，并自动将 `allgather dispatcher` 切换为 `alltoall`（allgather 不支持变长序列）

**推理侧：SGLang（非 vLLM）**，SGLang 支持 EP 分组推理。

**RL 权重同步阶段的 EP 通信**：是真实瓶颈，但已针对性优化。
同步路径为两阶段异步流水（`update_weight_from_distributed.py:195-232`）：
1. TP all-gather（单机内）
2. EP all-gather（`async_op=True`，`dist.all_gather(..., group=mpu.get_expert_model_parallel_group())`）→ HF 格式转换 → broadcast 到 SGLang

框架用 `update_weight_buffer_size` 控制 bucket 攒批缓解，但跨节点 EP all-gather 在 256 专家规模下是不可消除的通信开销。

**122B RL 闭环验证：未验证**。框架在 35B-A3B（同架构 MoE+GDN）上有完整训练脚本，122B-A10B 属于规模外推，需实测确认收敛。

**最大 Sequence Length**：无硬编码上限，由显存决定。训练侧用 `max_tokens_per_gpu` 动态 batch。参考值：35B-A3B 多模态配置 6144 tokens/GPU（8xGPU），122B-A10B 激活参数约 3.5x 大，同等显存下预计 **2048–4096 tokens/GPU**，需实测。

#### 2. Trajectory Packing 与长序列计算

**变长序列（varlen）：强制开启，完全无 padding 计算**
- `relax/backends/megatron/arguments.py:38`：`args.variable_seq_lengths = True`，无条件强制，所有训练任务默认走变长路径
- `relax/backends/megatron/data.py:162-203`：构建 `cu_seqlens`，传入 `PackedSeqParams(cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, qkv_format="thd")`，直接对接 FlashAttention 变长算子，**完全无 padding 开销**
- CP（Context Parallel）下已处理全局 pad 到 `cp_size` 整数倍的边界对齐

**极端 length imbalance**：`use_dynamic_batch_size` + `max_tokens_per_gpu` 按 token 总数打包，天然缓解长度不齐，但 trajectory-level packing（多轮轨迹拼长序列）需要精确的 `loss_mask` 标注（见下文）。

#### 3. 多模态 Token 混合处理

**支持，但依赖 HF processor，且 122B 多模态 RL 未端到端验证**
- `relax/utils/multimodal/`：完整 image/video/audio 处理基础设施
- Rollout 侧：图像 base64 编码后传入 SGLang，SGLang 原生支持 vision token 插入
- 训练侧：`pixel_values`、`image_grid_thw` 作为额外 forward kwargs 传入，M-RoPE 由 Bridge 处理
- 框架不自己做 image token → patch embedding 映射，完全委托给 HF processor + SGLang

---

### 模块二：多轮 Agentic 核心机制

#### 1. 异步 Rollout 与策略同步

**环境异步挂起**：SGLang 引擎支持连续 batch，一个请求等待 I/O 时调度其他请求。框架层的 `fully_async` 指训练和 rollout 的流水线异步，不是单次生成内的挂起。`on_policy_distillation.py` 用 `aiohttp` 异步调用 teacher，IO 等待在 async 事件循环内处理，不阻塞其他 rollout。

**Off-policy 容忍度与 IS 修正**：
- `max_staleness`（`relax/components/rollout.py:293-310`）：控制允许的陈旧度（rollout_id 差值），超过阈值的 stale data **直接丢弃，不做 IS 修正**
- OPSM（Off-Policy Sequence Masking，`relax/utils/training/ppo_utils.py:55-92`）：对 advantage < 0 且序列级 KL > delta 的样本做整体 mask，是保守过滤，**不是加权 IS 修正**
- `compute_approx_kl` 有 `importance_ratio` 参数（`ppo_utils.py:18-46`），基础设施存在，但未在主流程中接入完整 IS 链路
- **结论：有工程级 off-policy 过滤，没有完整的重要性采样修正（无 V-trace / PPO-style IS clipping）**

#### 2. On-Policy Distillation (OPD)

**Teacher 部署：支持两种模式**
- SGLang HTTP teacher（`relax/engine/rollout/on_policy_distillation.py`，通过 `aiohttp` 异步请求外部推理服务）
- Megatron reference actor（框架内部调度）
- 显存隔离通过 Ray resource allocation 实现，可将 teacher 放在独立 GPU 组

**Action-level logits 打分**：
- `Sample.teacher_log_probs: list[float]`（`relax/utils/types.py:31`）：per-token log probability
- `Sample.teacher_topk_token_ids: list[list[int]]`（`types.py:32`）：top-k token IDs
- 数据结构支持 token 粒度打分，但**对中间特定 Action 步骤的标注需在 custom rollout function 中自行实现**（通过 `loss_mask` 指定 action span）

#### 3. GRPO 多轮优势计算与 Loss Masking

**Credit Assignment：仅支持 episode-level scalar，无 step-level**
- `relax/components/advantages.py:148-151`：`rewards` 是 `list[float]`（episode 级标量），`get_grpo_returns` 做组内归一化后展开到所有 token
- **多轮场景下，中间每步的 advantage 与最终 episode reward 相同，无 step-level / return-to-go 分配**
- PPO estimator 支持 GAE 逐步递推，但也依赖 step-level reward 输入

**Jagged Masking：接口存在，无自动多轮工具**
- `Sample.loss_mask: list[int]`（`types.py:25`）：token 级别二值 mask，rollout 侧和训练侧均有精确对齐处理
- **框架没有提供多轮自动生成 action-only mask 的工具函数**，需在 custom rollout function 中自行实现多轮交替的 Action token 标注

#### 4. 密稀结合的奖励注入

**数据结构支持 dict reward，但 advantage 计算链路只消费 episode scalar**
- `Sample.reward: float | dict[str, Any] | None`（`types.py:24`）：结构上支持多 key reward dict
- `RewardWorker.compute`（`relax/engine/rewards/__init__.py:61`）：各 rm_type 返回单个 float
- `relax/components/advantages.py`：`rewards: list[float]` 输入，**不消费 turn-level 或 token-level 中间奖励**
- `Sample.train_metadata: dict`（`types.py:49`）可携带任意元信息
- **结论：注入 turn-level dense reward 需修改 `advantages.py` 的输入和计算逻辑，框架层当前不支持**

---

### 能力总表

| 问题 | 结论 | 核心约束 / 关键代码 |
|------|------|---------------------|
| EP 支持（训练） | 支持 | 权重同步 EP all-gather 是真实瓶颈；`update_weight_from_distributed.py:195-232` |
| EP 支持（推理） | 支持（SGLang） | 非 vLLM |
| 122B RL 闭环验证 | 未验证 | 需实测 |
| 最大 Sequence Length | 无硬限，显存决定 | 122B 估算 2048–4096 tokens/GPU |
| Trajectory Packing | 完全支持，默认开启 | `arguments.py:38`；`data.py:162-203` cu_seqlens + FlashAttention |
| 多模态 Token 混合 | 支持，依赖 HF processor | 122B 多模态 RL 未端到端验证 |
| 环境交互异步调度 | 支持 | SGLang 连续 batch；框架 `fully_async` 流水线；`aiohttp` 事件循环 |
| Off-policy IS 修正 | 无完整 IS | 有 OPSM 过滤（丢弃），无重要性采样加权；`ppo_utils.py:55-92` |
| OPD Teacher 调度 | 支持 | HTTP or 内部 Megatron；token 粒度 logprob；action span 标注需自定义 |
| 多轮 step-level advantage | 不支持 | GRPO 只有 episode-level scalar reward；`advantages.py:148-151` |
| Jagged Action Masking | 接口支持，无工具 | `loss_mask` 字段存在，无自动多轮 action mask 生成；`types.py:25` |
| 中间 Dense Reward 注入 | 不支持（需改代码） | reward dict 结构存在，advantage 链路不消费；`advantages.py` 需修改 |

---

## 待解决问题

- 128 卡配置下的并行策略（TP/PP/EP）：尚未确定（参考：35B-A3B 用 TP=2, EP=8, 8xGPU）
- MTP 层处理：config 中 `mtp_num_hidden_layers=1`，训练脚本是否需要 `--enable-mtp-training`
- Vision encoder 冻结策略

---

## 后续计划

1. ~~新建 `scripts/models/qwen35-122B-A10B.sh`~~ ✅ 已完成
2. ~~验证权重转换逻辑~~ ✅ 71/71 通过
3. 新建 `scripts/training/multimodal/run-qwen35-122B-A10B-{N}xgpu-async.sh`
4. 确认 122B-A10B 的并行策略（TP × PP × EP × 节点数 = 总 GPU 数）
