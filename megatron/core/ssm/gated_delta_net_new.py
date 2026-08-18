# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_global_memory_buffer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.layers import dist_all_gather_func, dist_reduce_scatter_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    cat_with_oom_fallback,
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None

    HAVE_FLA = False

logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNet(MegatronModule):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        name: str | None = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
            name (str | None): module instance name passed top-down from its paranet module
        """

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # The in_proj / out_proj weights are CP-sliced by hand (see `get_parameter_local_cp`)
        # and applied via a raw `F.linear` in `forward`, rather than calling
        # `self.in_proj` / `self.out_proj` directly. This means their forward() is never
        # invoked, so TE's delayed weight-gradient computation (which relies on state set up
        # during that forward call) is not available here; weight grads are instead computed
        # synchronously by standard autograd. Fail fast if delayed wgrad is requested.
        assert not config.delay_wgrad_compute, (
            "GatedDeltaNet's CP+SP fused linear path bypasses in_proj/out_proj's own "
            "forward(), so delay_wgrad_compute is not supported for this module."
        )

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
            name=(name + ".in_proj") if name is not None else None,
        )

        # Conv1d for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        if self.config.deterministic_mode:
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        else:
            self.gated_delta_rule = chunk_gated_delta_rule

        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
            name=(name + ".out_proj") if name is not None else None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                torch.ones(
                    self.num_v_heads_local_tp,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # print(f"packed_seq_params: {packed_seq_params}, shape: {hidden_states.shape}, cp_size: {self.cp_size}, sp_size: {self.sp_size}")
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

            # Resolve cu_seqlens with alignment padding handling.
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len,
                "cu_seqlens_kv",
                cp_size=self.cp_size,
            )
            assert torch.equal(cu_seqlens_q, cu_seqlens_kv), (
                "Currently only support cu_seqlens_q equals to cu_seqlens_kv, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
            num_packed_seqs = cu_seqlens_q.shape[0] - 1
            assert num_packed_seqs > 0, (
                "Number of packed sequences must be greater than 0, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None


        # ============================================================
        # Stage 1: CP+SP Column Parallel Linear (in_proj)
        # Gathers hidden_states across the combined SP+CP domain, then applies a local
        # matmul with the in_proj weight further sliced along CP (on top of its existing
        # TP shard). Output is still in CP zig-zag (load-balanced) sequence order.
        # ============================================================
        nvtx_range_push(suffix="in_proj")
        qkvzba = column_parallel_linear_cp_sp(
            hidden_states,
            self.in_proj.weight,
            self.in_proj.bias if self.bias else None,
            cp_group=self.pg_collection.cp,
            tp_group=self.pg_collection.tp,
            tp_cp_group=self.pg_collection.tp_cp,
            sp_enabled=self.sp_size > 1,
            split_sections=[
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
        )
        nvtx_range_pop(suffix="in_proj")

        # ============================================================
        # Stage 2: Undo CP zig-zag (attention load balancing) ordering.
        # Converts the CP-gathered sequence from zig-zag order to natural sequential
        # order so that the causal convolution and gated delta rule below can process
        # the tokens causally. Backward redoes the zig-zag ordering.
        # ============================================================
        nvtx_range_push(suffix="undo_zigzag")
        qkvzba = undo_zigzag_cp(qkvzba, self.cp_size, packed_seq_params)
        nvtx_range_pop(suffix="undo_zigzag")

        # ============================================================
        # Stage 3: middle computation (convolution + gated delta rule), see below.
        # ============================================================

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,  # Torch-native only accept [b, d, s] format input
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in ["silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv,  # FLA conv1d accepts [b, s, d] format input
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="conv1d")

        # Prepare QKV tensors (split, reshape, L2 norm, repeat_interleave, contiguous)
        nvtx_range_push(suffix="prepare_qkv_for_gated_delta_rule")
        query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
            qkv, gate, beta, alpha, batch, seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_gated_delta_rule")

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )
        g, beta = self._compute_g_and_beta(A_log_local_cp, dt_bias_local_cp, alpha, beta)
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, last_recurrent_state = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # ============================================================
        # Stage 4: Redo CP zig-zag (attention load balancing) ordering.
        # Converts the natural-order sequence produced by the middle computation back
        # to zig-zag order, mirroring Stage 2. Backward undoes the zig-zag ordering.
        # ============================================================
        nvtx_range_push(suffix="redo_zigzag")
        norm_out = redo_zigzag_cp(norm_out, self.cp_size, packed_seq_params)
        nvtx_range_pop(suffix="redo_zigzag")

        # ============================================================
        # Stage 5: CP+SP Row Parallel Linear (out_proj)
        # Applies a local matmul with the out_proj weight further sliced along CP (on
        # top of its existing TP shard), then reduces (sums) the CP+TP partial outputs
        # and scatters the result back to the local SP+CP sequence shard.
        # ============================================================
        nvtx_range_push(suffix="out_proj")
        out, out_bias = row_parallel_linear_cp_sp(
            norm_out,
            self.out_proj.weight,
            self.out_proj.bias if self.bias else None,
            cp_group=self.pg_collection.cp,
            tp_group=self.pg_collection.tp,
            tp_cp_group=self.pg_collection.tp_cp,
            sp_enabled=self.sp_size > 1,
        )
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        """
        Prepare query, key, value, gate, beta, alpha tensors for gated delta rule.
        Fuses split, reshape, L2 norm, repeat_interleave, and contiguous operations.
        """
        # Split qkv into query_key and value
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )

        # Reshape query_key and value
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        # Split query and key
        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        # Expand query and key if needed (grouped query attention)
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # Make all tensors contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        return query, key, value, gate, beta, alpha

    @jit_fuser
    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        """
        Compute g (decay) and beta (sigmoid) for gated delta rule.
        Fuses exp, softplus, mul, neg, and sigmoid operations.
        """
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        return g, beta

    def _resolve_cu_seqlens(
        self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1
    ):
        """Resolve cu_seqlens for packed sequence all-to-all, handling alignment padding."""
        if cu_seqlens_padded is not None:
            cu_seqlens = cu_seqlens_padded
        else:
            cu_seqlens = cu_seqlens_actual

        # total_cu = cu_seqlens[-1].cpu().item()
        # if total_cu != total_seq_len:
        #     raise ValueError(
        #         f"GDN: {name}[-1]={total_cu} does not match "
        #         f"total_sequence_length={total_seq_len}. "
        #         f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
        #     )

        # seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # if (seq_lengths % cp_size != 0).any():
        #     raise ValueError(
        #         f"All per-sequence lengths in cu_seqlens must be divisible by cp_size={cp_size}, "
        #         f"but got lengths: {seq_lengths.tolist()}"
        #     )

        return cu_seqlens

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {f"weight": 0}
                if self.conv_bias:
                    tp_sharding_map[f"bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        """No-op: weight gradients for in_proj/out_proj are already computed synchronously.

        `forward` never calls `self.in_proj`/`self.out_proj` directly — it takes a
        differentiable CP-local slice of their weight/bias (see `get_parameter_local_cp`)
        and applies it via a plain `F.linear` in `column_parallel_linear_cp_sp` /
        `row_parallel_linear_cp_sp`. Standard autograd therefore already accumulates the
        weight/bias gradients onto the original parameters during the main backward pass,
        so there is no deferred weight-gradient computation left to perform here (see the
        `delay_wgrad_compute` assertion in `__init__`).
        """

    def _backward_in_proj(self):
        """Deprecated: kept for API compatibility. See `backward_dw`."""

    def _backward_out_proj(self):
        """Deprecated: kept for API compatibility. See `backward_dw`."""


def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


####################
# Sharded state dict utilities
####################
def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        orig_sh_ten.replica_id,
    )


####################
# Context parallel utilities
####################
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_cp2hp.
    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    # Undo attention load balancing last if needed.
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_hp2cp.
    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_hp2cp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Redo attention load balancing first if needed.
    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


####################
# CP+SP fused Column/Row Parallel Linear
####################
def column_parallel_linear_cp_sp(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
    sp_enabled: bool,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Column parallel linear whose input activation is jointly sharded along the sequence
    dimension by both sequence-parallel (SP, within the TP group) and context-parallel (CP).

    All SP/CP communication (the gather itself, plus the TP all-reduce that would otherwise
    be a separate `copy_to_tensor_model_parallel_region` call when SP is disabled) is handled
    inside :class:`_ColumnParallelLinearCPSP`, so this wrapper only needs to CP-slice
    ``weight``/``bias`` before dispatching to it. See that class for the forward/backward
    communication details.

    Args:
        hidden_states (torch.Tensor): Local input activation shard,
            shape ``[seq_local, batch, hidden]``.
        weight (torch.Tensor): The TP-local weight (not yet CP-sliced), e.g. ``self.in_proj.weight``.
        bias (Optional[torch.Tensor]): The TP-local bias (not yet CP-sliced), or None.
        cp_group (torch.distributed.ProcessGroup): The context parallel process group.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel process group.
        tp_cp_group (torch.distributed.ProcessGroup): The combined tensor+context parallel
            process group (used only when ``sp_enabled``).
        sp_enabled (bool): Whether sequence-parallelism is enabled (i.e. ``sp_size > 1``).
        split_sections (Optional[List[int]]): Passed through to `get_parameter_local_cp` to
            slice ``weight``/``bias`` per logical section (e.g. q/k/v/z/beta/alpha) before
            taking the local CP chunk of each.

    Returns:
        torch.Tensor: Output activation, shape ``[seq_global, batch, out_features_local_tpcp]``,
            still in CP zig-zag sequence order.
    """
    weight_local_cp = get_parameter_local_cp(
        weight, dim=0, cp_group=cp_group, split_sections=split_sections
    )
    bias_local_cp = (
        get_parameter_local_cp(bias, dim=0, cp_group=cp_group, split_sections=split_sections)
        if bias is not None
        else None
    )

    return _ColumnParallelLinearCPSP.apply(
        hidden_states, weight_local_cp, bias_local_cp, cp_group, tp_group, tp_cp_group, sp_enabled
    )


_GDN_CP_SP_MEMORY_BUFFER_NAME = "gdn_column_parallel_cp_sp"


def _cp_sp_all_gather_into_buffer(
    input_: torch.Tensor, group: torch.distributed.ProcessGroup, async_op: bool = False
):
    """Raw (non-autograd) all-gather along the first dim into a reusable global buffer.

    Mirrors the buffer-reuse pattern used by `LinearWithGradAccumulationAndAsyncCommunication`
    (megatron/core/tensor_parallel/layers.py) to avoid a fresh allocation on every call. Callers
    must fully consume the returned tensor (e.g. via a matmul into a new tensor) before issuing
    another all-gather that reuses the same buffer name.
    """
    dim_size = list(input_.size())
    dim_size[0] *= group.size()
    buffer = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, _GDN_CP_SP_MEMORY_BUFFER_NAME)
    handle = dist_all_gather_func(buffer, input_.contiguous(), group=group, async_op=async_op)
    return buffer, handle


def _cp_sp_reduce_scatter_raw(
    grad_total: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    local_shape,
    async_op: bool = False,
):
    """Raw (non-autograd) reduce-scatter along the first dim."""
    sub_grad = torch.empty(
        local_shape, dtype=grad_total.dtype, device=grad_total.device, requires_grad=False
    )
    handle = dist_reduce_scatter_func(sub_grad, grad_total.contiguous(), group=group, async_op=async_op)
    return sub_grad, handle


class _ColumnParallelLinearCPSP(torch.autograd.Function):
    """Autograd function: forward = (SP+)CP gather + local matmul; backward recomputes the
    gather (overlapped with the weight-only ``grad_input`` matmul) instead of saving the
    gathered activation. See :func:`column_parallel_linear_cp_sp` for the wrapper this backs.

    All SP/CP communication lives here:
        - If ``sp_enabled``, a single all-gather over ``tp_cp_group`` reconstructs the full
          sequence (SP and CP happen to share the exact same rank layout convention:
          contiguous TP shards nested within each CP rank's zig-zag chunk), avoiding a second
          collective.
        - Otherwise, the input is already replicated across TP ranks, so only ``cp_group``
          needs an all-gather in the forward pass; in the backward pass, the corresponding
          ``grad_input`` must additionally be all-reduced (summed) across ``tp_group`` —
          mirroring `_CopyToModelParallelRegion`'s backward — since the same (replicated)
          input contributed to the local matmul on every TP rank.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, cp_group, tp_group, tp_cp_group, sp_enabled):
        ctx.cp_group = cp_group
        ctx.tp_group = tp_group
        ctx.tp_cp_group = tp_cp_group
        ctx.sp_enabled = sp_enabled
        ctx.has_bias = bias is not None
        ctx.save_for_backward(input_, weight)

        gather_group = tp_cp_group if sp_enabled else cp_group
        if gather_group.size() > 1:
            total_input, _ = _cp_sp_all_gather_into_buffer(input_, gather_group)
        else:
            total_input = input_

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        cp_group, tp_group, tp_cp_group, sp_enabled = (
            ctx.cp_group,
            ctx.tp_group,
            ctx.tp_cp_group,
            ctx.sp_enabled,
        )
        gather_group = tp_cp_group if sp_enabled else cp_group

        handle = None
        if gather_group.size() > 1:
            # Recompute (re-gather) total_input asynchronously; overlapped with the grad_input
            # matmul below, which only depends on `weight`.
            total_input, handle = _cp_sp_all_gather_into_buffer(input_, gather_group, async_op=True)
        else:
            total_input = input_

        grad_total_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        total_input_2d = total_input.reshape(-1, total_input.shape[-1])
        grad_weight = grad_output_2d.t().matmul(total_input_2d)
        grad_bias = grad_output_2d.sum(dim=0) if ctx.has_bias else None

        if gather_group.size() > 1:
            grad_input, rs_handle = _cp_sp_reduce_scatter_raw(
                grad_total_input, gather_group, input_.shape, async_op=True
            )
            rs_handle.wait()
        else:
            grad_input = grad_total_input

        if not sp_enabled and tp_group.size() > 1:
            # `input_` was replicated (not sharded) across TP ranks, so its gradient must be
            # summed across TP — mirrors `_CopyToModelParallelRegion.backward`.
            grad_input = grad_input.contiguous()
            torch.distributed.all_reduce(grad_input, group=tp_group)

        return grad_input, grad_weight, grad_bias, None, None, None, None


def row_parallel_linear_cp_sp(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
    sp_enabled: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Row parallel linear whose output activation must be reduced (summed) across the
    combined SP+CP domain and re-sharded back to the local SP+CP sequence chunk.

    Forward:
        1. Slice ``weight`` (already TP-local, ``input_is_parallel=True`` style) further
           along CP (input feature dimension) and apply ``F.linear``. Every rank in the
           combined TP+CP domain now holds a *partial* contribution (over disjoint input
           feature ranges) to the same full-size, full-sequence output.
        2. Reduce (sum) the TP+CP partial contributions and scatter the result back along
           the sequence dimension to each rank's local SP+CP shard.
           - If SP is enabled, this is a single reduce-scatter over ``tp_cp_group``.
           - Otherwise, ``cp_group`` handles the reduce-scatter (sum + seq split) and
             ``tp_group`` handles a plain all-reduce (sum only, no seq split, since the
             sequence isn't SP-sharded across TP in this case).
        3. ``bias`` (if any) is *not* sharded — like ``self.out_proj`` (constructed with
           ``skip_bias_add=True``), it is returned unfused alongside ``output`` for the
           caller to add (e.g. fused with a residual add), mirroring Megatron's own
           ``RowParallelLinear`` bias handling.

    Unlike :func:`column_parallel_linear_cp_sp`, this does *not* need a fused
    gather-recompute autograd.Function: no collective runs *before* the local matmul here
    (``input_`` is already the full sequence, at its natural local size — exactly mirroring
    Megatron's own ``RowParallelLinear``, which also calls the plain
    `LinearWithGradAccumulationAndAsyncCommunication` with `sequence_parallel=False` and does
    the reduce/reduce-scatter as a separate step afterwards). So there is no oversized
    intermediate activation whose lifetime needs shortening, and the reduce-scatter/all-reduce
    ops used below (`reduce_scatter_to_sequence_parallel_region`,
    `reduce_from_tensor_model_parallel_region`) are already lightweight, standard autograd
    Functions with no extra tensors to save.

    Backward: standard autograd.

    Args:
        input_ (torch.Tensor): Full-sequence local input, shape
            ``[seq_global, batch, in_features_local_tpcp]``, in natural sequence order.
        weight (torch.Tensor): The TP-local weight (not yet CP-sliced), e.g. ``self.out_proj.weight``.
        bias (Optional[torch.Tensor]): The (un-sharded) bias, e.g. ``self.out_proj.bias``, or None.
        cp_group (torch.distributed.ProcessGroup): The context parallel process group.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel process group.
        tp_cp_group (torch.distributed.ProcessGroup): The combined tensor+context parallel
            process group (used only when ``sp_enabled``).
        sp_enabled (bool): Whether sequence-parallelism is enabled (i.e. ``sp_size > 1``).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Output activation, shape
            ``[seq_local, batch, out_features]``, and the unfused bias (or None).
    """
    weight_local_cp = get_parameter_local_cp(weight, dim=1, cp_group=cp_group)
    output = F.linear(input_, weight_local_cp)

    if sp_enabled:
        output = reduce_scatter_to_sequence_parallel_region(output, group=tp_cp_group)
    else:
        if cp_group.size() > 1:
            output = reduce_scatter_to_sequence_parallel_region(output, group=cp_group)
        output = reduce_from_tensor_model_parallel_region(output, group=tp_group)
    return output, bias


####################
# CP Zig-zag reordering (pure local reorder, no communication)
####################
def undo_zigzag_cp(
    tensor: torch.Tensor,
    cp_size: int,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> torch.Tensor:
    """Undo CP zig-zag (attention load balancing) ordering: converts a full-sequence tensor
    from zig-zag (load-balanced) order to natural sequential order, e.g. for cp_size=3,
    162534 -> 123456. Backward redoes the zig-zag ordering (the exact inverse permutation).

    Args:
        tensor (torch.Tensor): Full-sequence tensor in zig-zag order, shape
            ``[seq_global, batch, hidden]`` (SBD) or ``[total_tokens, 1, hidden]`` (TND/THD).
        cp_size (int): The context parallel world size.
        packed_seq_params (Optional[PackedSeqParams]): If not None and ``qkv_format == 'thd'``,
            the cu_seqlens from this object are used for per-sequence varlen reordering.

    Returns:
        torch.Tensor: Full-sequence tensor in natural (sequential) order, same shape as input.
    """
    if cp_size == 1:
        return tensor
    return _UndoZigZagCP.apply(tensor, cp_size, packed_seq_params)


def redo_zigzag_cp(
    tensor: torch.Tensor,
    cp_size: int,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> torch.Tensor:
    """Redo CP zig-zag (attention load balancing) ordering: converts a full-sequence tensor
    from natural sequential order to zig-zag (load-balanced) order, e.g. for cp_size=3,
    123456 -> 162534. This is the exact inverse of :func:`undo_zigzag_cp`. Backward undoes
    the zig-zag ordering.

    Args:
        tensor (torch.Tensor): Full-sequence tensor in natural order, shape
            ``[seq_global, batch, hidden]`` (SBD) or ``[total_tokens, 1, hidden]`` (TND/THD).
        cp_size (int): The context parallel world size.
        packed_seq_params (Optional[PackedSeqParams]): If not None and ``qkv_format == 'thd'``,
            the cu_seqlens from this object are used for per-sequence varlen reordering.

    Returns:
        torch.Tensor: Full-sequence tensor in zig-zag order, same shape as input.
    """
    if cp_size == 1:
        return tensor
    return _RedoZigZagCP.apply(tensor, cp_size, packed_seq_params)


class _UndoZigZagCP(torch.autograd.Function):
    """Autograd function: forward = undo CP zig-zag; backward = redo CP zig-zag."""

    @staticmethod
    def forward(ctx, tensor, cp_size, packed_seq_params):
        ctx.cp_size = cp_size
        ctx.packed_seq_params = packed_seq_params
        return _undo_attention_load_balancing(tensor, cp_size, packed_seq_params)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _redo_attention_load_balancing(
            grad_output, ctx.cp_size, ctx.packed_seq_params
        )
        return grad_input, None, None


class _RedoZigZagCP(torch.autograd.Function):
    """Autograd function: forward = redo CP zig-zag; backward = undo CP zig-zag."""

    @staticmethod
    def forward(ctx, tensor, cp_size, packed_seq_params):
        ctx.cp_size = cp_size
        ctx.packed_seq_params = packed_seq_params
        return _redo_attention_load_balancing(tensor, cp_size, packed_seq_params)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _undo_attention_load_balancing(
            grad_output, ctx.cp_size, ctx.packed_seq_params
        )
        return grad_input, None, None


####################
# Torch native gated delta rule
####################
def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    # pylint: disable=line-too-long
    '''
    Torch-native implementation of chunked gated delta rule for deterministic mode.
    Need this because FLA is not deterministic.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    '''

    assert (
        cu_seqlens is None
    ), "cu_seqlens is not supported for torch_chunk_gated_delta_rule for now."

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
