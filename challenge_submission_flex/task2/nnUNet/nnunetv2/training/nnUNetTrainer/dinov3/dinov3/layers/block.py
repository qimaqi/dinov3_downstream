# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import copy
import math
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
        

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn




class SelfAttentionBlockAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,

        # new adapter args
        use_adapter: bool = True,
        adapter_dim: int | None = None,
        adapter_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

        # adapters
        self.use_adapter = use_adapter
        self.adapter_scale = adapter_scale

        if use_adapter:
            print('Init Adapter of DINOV3')
            adapter_dim = adapter_dim if adapter_dim is not None else dim

            assert adapter_dim == dim, (
                "This implementation assumes Adapter input/output dim == dim. "
                "If adapter_dim != dim, add projection layers."
            )   

            self.Space_Inner_Adapter = Adapter(adapter_dim)  # with skip connection
            self.MLP_Inner_Adapter = Adapter(adapter_dim, skip_connect=False)
        else:
            self.Space_Inner_Adapter = nn.Identity()
            self.MLP_Inner_Adapter = nn.Identity()

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[Tensor, Tensor] | None,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            # print("Training or sample model", self.training, 'self.sample_drop_ratio', self.sample_drop_ratio)
            indices_1 = torch.randperm(b, device=x.device)[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)

            attn_in = self.norm1(x_subset_1)
            residual_1 = self.attn(attn_in, rope=rope_subset)
            residual_1 = self.Space_Inner_Adapter(residual_1)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = torch.randperm(b, device=x.device)[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            mlp_in = self.norm2(x_subset_2)

            residual_2 = self.mlp(mlp_in)

            if self.use_adapter:
                residual_2 = residual_2 + self.adapter_scale * self.MLP_Inner_Adapter(mlp_in)

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )

        else:
            attn_in = self.norm1(x)
            attn_out = self.attn(attn_in, rope=rope)
            attn_out = self.Space_Inner_Adapter(attn_out)

            x_attn = x + self.ls1(attn_out)

            mlp_in = self.norm2(x_attn)
            mlp_out = self.mlp(mlp_in)

            if self.use_adapter:
                mlp_out = mlp_out + self.adapter_scale * self.MLP_Inner_Adapter(mlp_in)

            x_ffn = x_attn + self.ls2(mlp_out)

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [
            max(int(b * (1 - self.sample_drop_ratio)), 1)
            for b in b_list
        ]
        residual_scale_factors = [
            b / sample_subset_size
            for b, sample_subset_size in zip(b_list, sample_subset_sizes)
        ]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                torch.randperm(b, device=x.device)[:sample_subset_size]
                for x, b, sample_subset_size in zip(
                    x_list, b_list, sample_subset_sizes
                )
            ]

            x_subset_1_list = [
                x[indices_1]
                for x, indices_1 in zip(x_list, indices_1_list)
            ]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1)
                    for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)

            residual_1_list = self.attn.forward_list(
                norm1,
                rope_list=rope_subset_list,
            )

            residual_1_list = [
                self.Space_Inner_Adapter(residual_1)
                for residual_1 in residual_1_list
            ]

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list,
                    residual_1_list,
                    indices_1_list,
                    residual_scale_factors,
                )
            ]

            indices_2_list = [
                torch.randperm(b, device=x.device)[:sample_subset_size]
                for x, b, sample_subset_size in zip(
                    x_list, b_list, sample_subset_sizes
                )
            ]

            x_subset_2_list = [
                x[indices_2]
                for x, indices_2 in zip(x_attn_list, indices_2_list)
            ]

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            if self.use_adapter:
                residual_2_list = [
                    residual_2 + self.adapter_scale * self.MLP_Inner_Adapter(norm2)
                    for residual_2, norm2 in zip(residual_2_list, norm2_list)
                ]

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list,
                    residual_2_list,
                    indices_2_list,
                    residual_scale_factors,
                )
            ]

        else:
            x_ffn = []

            for x, rope in zip(x_list, rope_list):
                attn_in = self.norm1(x)
                attn_out = self.attn(attn_in, rope=rope)
                attn_out = self.Space_Inner_Adapter(attn_out)

                x_attn = x + self.ls1(attn_out)

                mlp_in = self.norm2(x_attn)
                mlp_out = self.mlp(mlp_in)

                if self.use_adapter:
                    mlp_out = mlp_out + self.adapter_scale * self.MLP_Inner_Adapter(mlp_in)

                x_out = x_attn + self.ls2(mlp_out)
                x_ffn.append(x_out)

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list(
                [x_or_x_list],
                rope_list=[rope_or_rope_list],
            )[0]

        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]

            return self._forward_list(
                x_or_x_list,
                rope_list=rope_or_rope_list,
            )

        else:
            raise AssertionError



class BottleneckCrossSliceAttentionMixer(nn.Module):
    """Mix CLS/register tokens across slices with a zero-gated residual adapter."""

    def __init__(
        self,
        dim: int,
        num_global_tokens: int = 5,
        num_slices: int = 256,
        bottleneck_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        gate_init: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
    ) -> None:
        super().__init__()
        self.num_global_tokens = num_global_tokens
        self.use_slice_pos_embed = use_slice_pos_embed
        self.use_token_type_embed = use_token_type_embed
        self.slice_pos_embed = (
            nn.Parameter(torch.zeros(1, num_slices, 1, dim, device=device)) if use_slice_pos_embed else None
        )
        self.token_type_embed = (
            nn.Parameter(torch.zeros(1, 1, num_global_tokens, dim, device=device)) if use_token_type_embed else None
        )
        self.norm = norm_layer(dim)
        self.down = nn.Linear(dim, bottleneck_dim, device=device)
        self.norm_attn = norm_layer(bottleneck_dim)
        self.attn = nn.MultiheadAttention(
            bottleneck_dim,
            num_heads=num_heads,
            batch_first=True,
            device=device,
        )
        self.norm_mlp = norm_layer(bottleneck_dim)
        hidden_dim = int(bottleneck_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim, device=device),
            act_layer(),
            nn.Linear(hidden_dim, bottleneck_dim, device=device),
        )
        self.up = nn.Linear(bottleneck_dim, dim, device=device)
        self.gate = nn.Parameter(torch.full((1,), gate_init, device=device))

    def _add_pos_embed(self, x: Tensor) -> Tensor:
        _, d, t, _ = x.shape
        if self.slice_pos_embed is not None:
            if d > self.slice_pos_embed.shape[1]:
                raise ValueError(
                    f"Cross-slice mixer was initialized for at most {self.slice_pos_embed.shape[1]} slices, got {d}."
                )
            x = x + self.slice_pos_embed[:, :d]
        if self.token_type_embed is not None:
            if t > self.token_type_embed.shape[2]:
                raise ValueError(
                    f"Cross-slice mixer was initialized for at most {self.token_type_embed.shape[2]} global tokens, got {t}."
                )
            x = x + self.token_type_embed[:, :, :t]
        return x

    def forward(self, x: Tensor, D: Optional[int] = None, T: Optional[int] = None) -> Tensor:
        input_was_flat = x.ndim == 3
        if input_was_flat:
            if D is None:
                if T is None:
                    T = self.num_global_tokens
                if x.shape[1] % T != 0:
                    raise ValueError(f"Cannot infer slice count from flattened mixer input {tuple(x.shape)} with T={T}.")
                D = x.shape[1] // T
            if T is None:
                if x.shape[1] % D != 0:
                    raise ValueError(f"Cannot infer token count from flattened mixer input {tuple(x.shape)} with D={D}.")
                T = x.shape[1] // D
            x = x.reshape(x.shape[0], D, T, x.shape[-1])
        elif x.ndim != 4:
            raise ValueError(f"Expected cross-slice mixer input with shape (B, D, T, C) or (B, D*T, C), got {tuple(x.shape)}")

        x_input = x
        b, d, t, c = x.shape
        y = self._add_pos_embed(x).reshape(b, d * t, c)
        y = self.down(self.norm(y))
        y = y + self.attn(self.norm_attn(y), self.norm_attn(y), self.norm_attn(y), need_weights=False)[0]
        y = y + self.mlp(self.norm_mlp(y))
        delta = self.up(y).reshape(b, d, t, c)
        x = x_input + self.gate * delta
        if input_was_flat:
            x = x.reshape(b, d * t, c)
        return x


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: Optional[float] = None, device=None) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=device)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=device)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


class LoRAQKVLinear(nn.Module):
    """LoRA wrapper for packed qkv projections, adapting q and v only."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: Optional[float] = None, device=None) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(f"Expected packed qkv output dimension divisible by 3, got {base.out_features}.")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        dim = base.out_features // 3
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.lora_q_a = nn.Linear(base.in_features, rank, bias=False, device=device)
        self.lora_q_b = nn.Linear(rank, dim, bias=False, device=device)
        self.lora_v_a = nn.Linear(base.in_features, rank, bias=False, device=device)
        self.lora_v_b = nn.Linear(rank, dim, bias=False, device=device)
        nn.init.kaiming_uniform_(self.lora_q_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_b.weight)
        nn.init.zeros_(self.lora_v_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.base(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.lora_q_b(self.lora_q_a(x)) * self.scaling
        v = v + self.lora_v_b(self.lora_v_a(x)) * self.scaling
        return torch.cat((q, k, v), dim=-1)


class _CopiedTransformerCore(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.norm1 = copy.deepcopy(block.norm1)
        self.attn = copy.deepcopy(block.attn)
        self.ls1 = copy.deepcopy(block.ls1)
        self.norm2 = copy.deepcopy(block.norm2)
        self.mlp = copy.deepcopy(block.mlp)
        self.ls2 = copy.deepcopy(block.ls2)

    def forward(self, x: Tensor) -> Tensor:
        x_attn = x + self.ls1(self.attn(self.norm1(x), rope=None))
        return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))


class CrossSliceMixerBase(nn.Module):
    def _reshape_input(self, x: Tensor, D: Optional[int], T: Optional[int]) -> tuple[Tensor, bool, int, int, int, int]:
        input_was_flat = x.ndim == 3
        if input_was_flat:
            if D is None:
                if T is None:
                    T = self.num_global_tokens
                if x.shape[1] % T != 0:
                    raise ValueError(f"Cannot infer slice count from flattened mixer input {tuple(x.shape)} with T={T}.")
                D = x.shape[1] // T
            if T is None:
                if x.shape[1] % D != 0:
                    raise ValueError(f"Cannot infer token count from flattened mixer input {tuple(x.shape)} with D={D}.")
                T = x.shape[1] // D
            x = x.reshape(x.shape[0], D, T, x.shape[-1])
        elif x.ndim != 4:
            raise ValueError(
                f"Expected cross-slice mixer input with shape (B, D, T, C) or (B, D*T, C), got {tuple(x.shape)}"
            )
        b, d, t, c = x.shape
        return x, input_was_flat, b, d, t, c

    def _add_pos_embed(self, x: Tensor) -> Tensor:
        _, d, t, _ = x.shape
        if self.slice_pos_embed is not None:
            if d > self.slice_pos_embed.shape[1]:
                raise ValueError(
                    f"Cross-slice mixer was initialized for at most {self.slice_pos_embed.shape[1]} slices, got {d}."
                )
            x = x + self.slice_pos_embed[:, :d]
        if self.token_type_embed is not None:
            if t > self.token_type_embed.shape[2]:
                raise ValueError(
                    f"Cross-slice mixer was initialized for at most {self.token_type_embed.shape[2]} global tokens, got {t}."
                )
            x = x + self.token_type_embed[:, :, :t]
        return x


class LayerCopiedLoRACrossSliceMixer(CrossSliceMixerBase):
    """Cross-slice mixer initialized from the current DINO block with frozen base weights and LoRA adapters."""

    def __init__(
        self,
        dim: int,
        pretrained_block: nn.Module,
        num_global_tokens: int = 5,
        num_slices: int = 256,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        gate_init: float = 0.0,
        device=None,
    ) -> None:
        super().__init__()
        self.num_global_tokens = num_global_tokens
        self.slice_pos_embed = (
            nn.Parameter(torch.zeros(1, num_slices, 1, dim, device=device)) if use_slice_pos_embed else None
        )
        self.token_type_embed = (
            nn.Parameter(torch.zeros(1, 1, num_global_tokens, dim, device=device)) if use_token_type_embed else None
        )
        self.block = _CopiedTransformerCore(pretrained_block)
        self.block.to(device=device)
        for param in self.block.parameters():
            param.requires_grad = False

        if not hasattr(self.block.attn, "qkv"):
            raise ValueError("Layer-copied LoRA mixer expects the copied attention module to expose a qkv projection.")
        self.block.attn.qkv = LoRAQKVLinear(self.block.attn.qkv, rank=lora_rank, alpha=lora_alpha, device=device)
        if hasattr(self.block.mlp, "fc1") and hasattr(self.block.mlp, "fc2"):
            self.block.mlp.fc1 = LoRALinear(self.block.mlp.fc1, rank=lora_rank, alpha=lora_alpha, device=device)
            self.block.mlp.fc2 = LoRALinear(self.block.mlp.fc2, rank=lora_rank, alpha=lora_alpha, device=device)
        else:
            raise ValueError("Layer-copied LoRA mixer currently expects an MLP with fc1 and fc2 projections.")
        self.gate = nn.Parameter(torch.full((1,), gate_init, device=device))

    def forward(self, x: Tensor, D: Optional[int] = None, T: Optional[int] = None) -> Tensor:
        x, input_was_flat, b, d, t, c = self._reshape_input(x, D, T)
        x_input = x
        x_flat = self._add_pos_embed(x).reshape(b, d * t, c)
        out = self.block(x_flat)
        delta = out - x_flat
        x = x_input.reshape(b, d * t, c) + self.gate * delta
        if not input_was_flat:
            x = x.reshape(b, d, t, c)
        return x


class TokenOnlyMLPMixer(CrossSliceMixerBase):
    """Mix only across the flattened slice/global-token dimension."""

    def __init__(
        self,
        dim: int,
        num_global_tokens: int = 5,
        num_slices: int = 256,
        mlp_ratio: float = 4.0,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        gate_init: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
    ) -> None:
        super().__init__()
        self.num_global_tokens = num_global_tokens
        self.slice_pos_embed = (
            nn.Parameter(torch.zeros(1, num_slices, 1, dim, device=device)) if use_slice_pos_embed else None
        )
        self.token_type_embed = (
            nn.Parameter(torch.zeros(1, 1, num_global_tokens, dim, device=device)) if use_token_type_embed else None
        )
        max_tokens = num_slices * num_global_tokens
        hidden_dim = int(max_tokens * mlp_ratio)
        self.norm = norm_layer(dim)
        self.fc1_weight = nn.Parameter(torch.empty(hidden_dim, max_tokens, device=device))
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim, device=device))
        self.fc2_weight = nn.Parameter(torch.empty(max_tokens, hidden_dim, device=device))
        self.fc2_bias = nn.Parameter(torch.zeros(max_tokens, device=device))
        nn.init.kaiming_uniform_(self.fc1_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_weight, a=math.sqrt(5))
        self.act = act_layer()
        self.gate = nn.Parameter(torch.full((1,), gate_init, device=device))

    def forward(self, x: Tensor, D: Optional[int] = None, T: Optional[int] = None) -> Tensor:
        x, input_was_flat, b, d, t, c = self._reshape_input(x, D, T)
        x_input = x
        n = d * t
        if n > self.fc2_weight.shape[0]:
            raise ValueError(
                f"Token-only mixer was initialized for at most {self.fc2_weight.shape[0]} tokens, got {n}."
            )
        y = self._add_pos_embed(x).reshape(b, n, c)
        y = self.norm(y).transpose(1, 2)
        hidden = int(n * self.fc1_weight.shape[0] / self.fc1_weight.shape[1])
        y = F.linear(y, self.fc1_weight[:hidden, :n], self.fc1_bias[:hidden])
        y = self.act(y)
        y = F.linear(y, self.fc2_weight[:n, :hidden], self.fc2_bias[:n])
        delta = y.transpose(1, 2)
        x = x_input.reshape(b, n, c) + self.gate * delta
        if not input_was_flat:
            x = x.reshape(b, d, t, c)
        return x


class CrossSliceGlobalTokenMixer(nn.Module):
    """Unified wrapper for cross-slice global-token mixer ablations."""

    def __init__(
        self,
        mode: Optional[str],
        dim: int,
        num_global_tokens: int = 5,
        num_slices: int = 256,
        num_heads: int = 4,
        bottleneck_dim: int = 128,
        mlp_ratio: float = 2.0,
        lora_rank: int = 16,
        pretrained_block: Optional[nn.Module] = None,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        gate_init: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
    ) -> None:
        super().__init__()
        if isinstance(mode, str):
            mode = mode.replace("-", "_")
        if mode in (None, "none"):
            self.mixer = nn.Identity()
        elif mode == "bottleneck_attention":
            self.mixer = BottleneckCrossSliceAttentionMixer(
                dim=dim,
                num_global_tokens=num_global_tokens,
                num_slices=num_slices,
                bottleneck_dim=bottleneck_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_slice_pos_embed=use_slice_pos_embed,
                use_token_type_embed=use_token_type_embed,
                gate_init=gate_init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                device=device,
            )
        elif mode == "layer_copied_lora":
            if pretrained_block is None:
                raise ValueError("cross_slice_mixer='layer_copied_lora' requires a pretrained_block.")
            self.mixer = LayerCopiedLoRACrossSliceMixer(
                dim=dim,
                pretrained_block=pretrained_block,
                num_global_tokens=num_global_tokens,
                num_slices=num_slices,
                lora_rank=lora_rank,
                use_slice_pos_embed=use_slice_pos_embed,
                use_token_type_embed=use_token_type_embed,
                gate_init=gate_init,
                device=device,
            )
        elif mode in ("token_only_mlp_mixer", "mlp_mixer"):
            self.mixer = TokenOnlyMLPMixer(
                dim=dim,
                num_global_tokens=num_global_tokens,
                num_slices=num_slices,
                mlp_ratio=mlp_ratio,
                use_slice_pos_embed=use_slice_pos_embed,
                use_token_type_embed=use_token_type_embed,
                gate_init=gate_init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported cross-slice mixer mode: {mode}")

    def forward(self, x: Tensor, D: Optional[int] = None, T: Optional[int] = None) -> Tensor:
        if isinstance(self.mixer, nn.Identity):
            return x
        return self.mixer(x, D=D, T=T)


class SliceSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        cross_slice_mixer: Optional[str] = "mlp_mixer",
        num_global_tokens: int = 5,
        slice_mixer_num_slices: int = 256,
        slice_mixer_bottleneck_dim: int = 128,
        slice_mixer_num_heads: int = 4,
        slice_mixer_mlp_ratio: float = 2.0,
        slice_mixer_init_gamma: float = 0.0,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.num_global_tokens = num_global_tokens
        self.cross_slice_mixer = CrossSliceGlobalTokenMixer(
            mode=cross_slice_mixer,
            dim=dim,
            num_global_tokens=num_global_tokens,
            num_slices=slice_mixer_num_slices,
            num_heads=slice_mixer_num_heads,
            bottleneck_dim=slice_mixer_bottleneck_dim,
            mlp_ratio=slice_mixer_mlp_ratio,
            pretrained_block=self,
            use_slice_pos_embed=use_slice_pos_embed,
            use_token_type_embed=use_token_type_embed,
            gate_init=slice_mixer_init_gamma,
            act_layer=act_layer,
            norm_layer=norm_layer,
            device=device,
        )

    def _mix_cross_slice_global_tokens(self, x: Tensor) -> Tensor:
        if isinstance(self.cross_slice_mixer.mixer, nn.Identity) or self.num_global_tokens <= 0:
            return x
        if x.shape[1] < self.num_global_tokens:
            raise ValueError(
                f"Expected at least {self.num_global_tokens} global tokens before patch tokens, got {x.shape[1]}."
            )
        global_tokens = x[:, : self.num_global_tokens].unsqueeze(0)
        mixed_global_tokens = self.cross_slice_mixer(global_tokens).squeeze(0)
        return torch.cat((mixed_global_tokens, x[:, self.num_global_tokens :]), dim=1)

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        print("x shape", x.shape)
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return self._mix_cross_slice_global_tokens(x_ffn)

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return [self._mix_cross_slice_global_tokens(x) for x in x_ffn]

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError

