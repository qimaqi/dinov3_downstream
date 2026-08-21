# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN

try:
    from dinov3.layers import SelfAttentionBlockAdapter
except:
    # raise NotImplementedError('No SelfAttentionBlockAdapter')
    print("SelfAttentionBlockAdapter not found")
    SelfAttentionBlockAdapter = SelfAttentionBlock

from dinov3.layers import SliceSelfAttentionBlock

from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        use_adapter=False,
        block_fn: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        block_cls = block_fn or SelfAttentionBlock
        if use_adapter:
            print("Using Adapter of DINOV3")
            blocks_list = [
                SelfAttentionBlockAdapter(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio_sequence[i],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for i in range(depth)
            ]
        else:
            blocks_list = [
                block_cls(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio_sequence[i],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for i in range(depth)
            ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_small_adapter(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        use_adapter=True,
        **kwargs,
    )
    return model



def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base_pretrain_lvd1689m(
    patch_size=16,
    weights: str | None = None,
    strict: bool = True,
    **kwargs,
):
    """
    ViT-B/16 config matching dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth.

    Pass weights to load a local checkpoint path with the same architecture as
    the official DINOv3 LVD1689M ViT-B/16 pretrained backbone.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        qkv_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        **kwargs,
    )
    if weights is not None:
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=strict)
    return model


def vit_base_adapter(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        use_adapter=True,
        **kwargs,
    )
    return model



def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model






# video version



class SliceWiseDinoVisionTransformer(DinoVisionTransformer):
    """Apply the 2D DINO ViT independently to slices from each 5D volume item."""

    def __init__(
        self,
        *,
        slice_axis: int = 2,
        cross_slice_mixer: Optional[str] = "mlp_mixer",
        slice_mixer_num_slices: int = 256,
        slice_mixer_bottleneck_dim: int = 128,
        slice_mixer_num_heads: int = 4,
        slice_mixer_mlp_ratio: float = 2.0,
        slice_mixer_init_gamma: float = 0.0,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
        **kwargs,
    ):
        if kwargs.get("use_adapter", False):
            raise ValueError(
                "SliceWiseDinoVisionTransformer uses SliceSelfAttentionBlock and does not support use_adapter=True."
            )
        num_global_tokens = kwargs.get("n_storage_tokens", 0) + 1
        kwargs["block_fn"] = partial(
            SliceSelfAttentionBlock,
            cross_slice_mixer=cross_slice_mixer,
            num_global_tokens=num_global_tokens,
            slice_mixer_num_slices=slice_mixer_num_slices,
            slice_mixer_bottleneck_dim=slice_mixer_bottleneck_dim,
            slice_mixer_num_heads=slice_mixer_num_heads,
            slice_mixer_mlp_ratio=slice_mixer_mlp_ratio,
            slice_mixer_init_gamma=slice_mixer_init_gamma,
            use_slice_pos_embed=use_slice_pos_embed,
            use_token_type_embed=use_token_type_embed,
        )
        super().__init__(**kwargs)
        self.slice_axis = self._normalize_slice_axis(slice_axis)

    @staticmethod
    def _normalize_slice_axis(slice_axis: int) -> int:
        if slice_axis < 0:
            slice_axis += 5
        if slice_axis not in (2, 3, 4):
            raise ValueError(
                f"slice_axis must refer to D/H/W in a (B, C, D, H, W) tensor, got {slice_axis}."
            )
        return slice_axis

    def _flatten_item_to_slices(self, x_item: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        # print("x_item", x_item.shape)
        if x_item.ndim != 4:
            raise ValueError(f"Expected one volume item with shape (C, D, H, W), got {tuple(x_item.shape)}")
        spatial_axes = [2, 3, 4]
        in_plane_axes = [axis for axis in spatial_axes if axis != self.slice_axis]
        item_slice_axis = self.slice_axis - 1
        item_in_plane_axes = [axis - 1 for axis in in_plane_axes]
        x_slices = x_item.permute(item_slice_axis, 0, *item_in_plane_axes).contiguous()
        # print("x_slices", x_slices.shape)
        s, c, h, w = x_slices.shape
        meta = {
            "num_slices": s,
            "slice_axis": self.slice_axis,
            "in_plane_axes": in_plane_axes,
        }
        return x_slices.reshape(s, c, h, w), meta

    def _slice_masks_for_item(
        self,
        masks: Optional[Tensor],
        batch_idx: int,
        batch_size: int,
        num_slices: int,
    ) -> Optional[Tensor]:
        if masks is None:
            return None
        if masks.ndim == 2 and masks.shape[0] == batch_size * num_slices:
            start = batch_idx * num_slices
            return masks[start:start + num_slices]
        if masks.ndim in (3, 4) and masks.shape[:2] == (batch_size, num_slices):
            return masks[batch_idx].reshape(num_slices, -1)
        raise ValueError(
            "For 5D slice-wise input, masks must have shape (B*S, N), (B, S, N), "
            f"or (B, S, H, W); got {tuple(masks.shape)}."
        )

    @staticmethod
    def _stack_batch_outputs(items: List[Tensor]) -> Tensor:
        return torch.stack(items, dim=0).contiguous()

    def _restore_item_feature_map(self, x: Tensor, meta: Dict[str, Any]) -> Tensor:
        s = meta["num_slices"]
        in_plane_axes = meta["in_plane_axes"]
        slice_axis = meta["slice_axis"]
        if x.ndim != 4:
            raise ValueError(f"Expected per-item feature map (S, C, H, W), got {tuple(x.shape)}")
        if x.shape[0] != s:
            raise ValueError(f"Expected {s} slices, got output shape {tuple(x.shape)}")
        x = x.reshape(s, *x.shape[1:])
        dim_for_axis = {slice_axis: 0, in_plane_axes[0]: 2, in_plane_axes[1]: 3}
        return x.permute(1, dim_for_axis[2], dim_for_axis[3], dim_for_axis[4]).contiguous()

    def _restore_intermediate_item(self, x: Tensor, meta: Dict[str, Any], reshape: bool) -> Tensor:
        if reshape:
            return self._restore_item_feature_map(x, meta)
        return x

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            return super().forward_features(x, masks)

        batch_size = x.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            x_slices, meta = self._flatten_item_to_slices(x[batch_idx])
            masks_slices = self._slice_masks_for_item(masks, batch_idx, batch_size, meta["num_slices"])
            outputs.append(dict(super().forward_features(x_slices, masks_slices)))

        out = {}
        for key in ("x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm"):
            out[key] = self._stack_batch_outputs([item[key] for item in outputs])
        out["masks"] = None
        if any(item.get("masks") is not None for item in outputs):
            out["masks"] = self._stack_batch_outputs([item["masks"] for item in outputs])
        return out

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        if x.ndim != 5:
            return super().get_intermediate_layers(
                x,
                n=n,
                reshape=reshape,
                return_class_token=return_class_token,
                return_extra_tokens=return_extra_tokens,
                norm=norm,
            )

        batch_outputs = []
        metas = []
        for batch_idx in range(x.shape[0]):
            x_slices, meta = self._flatten_item_to_slices(x[batch_idx])
            metas.append(meta)
            batch_outputs.append(
                super().get_intermediate_layers(
                    x_slices,
                    n=n,
                    reshape=reshape,
                    return_class_token=return_class_token,
                    return_extra_tokens=return_extra_tokens,
                    norm=norm,
                )
            )

        num_outputs = len(batch_outputs[0])
        restored = []
        for output_idx in range(num_outputs):
            items = [sample_outputs[output_idx] for sample_outputs in batch_outputs]
            if not return_class_token and not return_extra_tokens:
                restored.append(
                    self._stack_batch_outputs([
                        self._restore_intermediate_item(item, metas[batch_idx], reshape)
                        for batch_idx, item in enumerate(items)
                    ])
                )
            elif return_class_token and not return_extra_tokens:
                patch_items, class_items = zip(*items)
                restored.append(
                    (
                        self._stack_batch_outputs([
                            self._restore_intermediate_item(item, metas[batch_idx], reshape)
                            for batch_idx, item in enumerate(patch_items)
                        ]),
                        self._stack_batch_outputs(list(class_items)),
                    )
                )
            elif not return_class_token and return_extra_tokens:
                patch_items, extra_items = zip(*items)
                restored.append(
                    (
                        self._stack_batch_outputs([
                            self._restore_intermediate_item(item, metas[batch_idx], reshape)
                            for batch_idx, item in enumerate(patch_items)
                        ]),
                        self._stack_batch_outputs(list(extra_items)),
                    )
                )
            else:
                patch_items, class_items, extra_items = zip(*items)
                restored.append(
                    (
                        self._stack_batch_outputs([
                            self._restore_intermediate_item(item, metas[batch_idx], reshape)
                            for batch_idx, item in enumerate(patch_items)
                        ]),
                        self._stack_batch_outputs(list(class_items)),
                        self._stack_batch_outputs(list(extra_items)),
                    )
                )
        return tuple(restored)



class MedDINOv3(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed_2D = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embed_3D = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        # Optional, but very helpful:
        assert embed_dim % (6 * num_heads) == 0, \
            f"embed_dim ({embed_dim}) must be divisible by 6*num_heads ({6*num_heads}) for 3D RoPE"

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed_2D = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        self.rope_embed_3D = RopePositionEmbedding3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed_2D._init_weights()
        self.rope_embed_3D._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        if x.dim() == 5:
            x = self.patch_embed_3D(x)
            B, D, H, W, _ = x.shape
            x = x.flatten(1, 3)
        else:
            x = self.patch_embed_2D(x)
            B, H, W,  _ = x.shape
            D = None
            x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, ((D, H, W) if D is not None else (H, W))

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
            
        if len(rope[0]) == 3:
            rope_sincos = [self.rope_embed_3D(D=D, H=H, W=W) for D, H, W in rope]
        else:
            rope_sincos = [self.rope_embed_2D(H=H, W=W) for H, W in rope]
                
        for _, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos)
            
        n_storage_tokens = self.n_storage_tokens
        norm = self.norm
        cls_norm = self.cls_norm
        local_cls_norm = self.local_cls_norm
        untie_cls_and_patch_norms = self.untie_cls_and_patch_norms
        untie_global_and_local_cls_norm = self.untie_global_and_local_cls_norm
        training = self.training
    
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if untie_cls_and_patch_norms or untie_global_and_local_cls_norm:
                if untie_global_and_local_cls_norm and training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = local_cls_norm(x[:, : n_storage_tokens + 1])
                elif untie_cls_and_patch_norms:
                    x_norm_cls_reg = cls_norm(x[:, : n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = norm(x[:, : n_storage_tokens + 1])
                x_norm_patch = norm(x[:, n_storage_tokens + 1 :])
            else:
                x_norm = norm(x)
                x_norm_cls_reg = x_norm[:, : n_storage_tokens + 1]
                x_norm_patch = x_norm[:, n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, hw_tuple = self.prepare_tokens_with_masks(x)
        if len(hw_tuple) == 3:
            D, H, W = hw_tuple
            rope_sincos = self.rope_embed_3D(D=D, H=H, W=W)
        else:
            H, W = hw_tuple
            rope_sincos = self.rope_embed_2D(H=H, W=W)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            if x.dim() == 5:
                B, _, d, h, w = x.shape
                outputs = [
                    out.reshape(B, d // self.patch_size, h // self.patch_size, w // self.patch_size, -1).permute(0, 4, 1, 2, 3).contiguous()
                    for out in outputs
                ]
            else:
                B, _, h, w = x.shape
                outputs = [
                    out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    @torch.no_grad()
    def inflate_patch_embed3d_from_2d(
        self,
        mode: str = "avg",   # "avg" or "center"
    ) -> None:
        """
        Initialize PatchEmbed3D weights by inflating PatchEmbed (2D) weights.

        Args:
            pe2d: 2D patch embed module with Conv2d `proj` of shape [C_out, C_in, kH, kW].
            pe3d: 3D patch embed module with Conv3d `proj` of shape [C_out, C_in, kD, kH, kW].
            mode:
                - "avg":   copy the 2D kernel into each temporal slice and divide by kD (I3D-style).
                - "center": copy into the center slice only; others set to 0.
        """
        assert isinstance(self.patch_embed_2D.proj, nn.Conv2d) and isinstance(self.patch_embed_3D.proj, nn.Conv3d), \
            "pe2d.proj must be Conv2d and pe3d.proj must be Conv3d"

        w2 = self.patch_embed_2D.proj.weight.data      # [C_out, C_in, kH2, kW2]
        b2 = self.patch_embed_2D.proj.bias.data if self.patch_embed_2D.proj.bias is not None else None

        w3 = self.patch_embed_3D.proj.weight.data      # [C_out, C_in, kD3, kH3, kW3]
        b3 = self.patch_embed_3D.proj.bias.data if self.patch_embed_3D.proj.bias is not None else None

        C_out2, C_in2, kH2, kW2 = w2.shape
        C_out3, C_in3, kD3, kH3, kW3 = w3.shape

        # Basic sanity checks
        assert C_out2 == C_out3, f"out_channels mismatch: {C_out2} vs {C_out3}"
        assert C_in2  == C_in3,  f"in_channels mismatch: {C_in2} vs {C_in3}"
        assert kH2    == kH3 and kW2 == kW3, \
            f"spatial kernel mismatch: (kH,kW)=({kH2},{kW2}) vs ({kH3},{kW3})"

        # Inflate: start from zeros
        w3.zero_()

        if mode == "avg":
            # Copy into every temporal slice and average across time
            # So the sum over temporal slices reproduces the 2D response.
            for t in range(kD3):
                w3[:, :, t, :, :].copy_(w2 / kD3)
        elif mode == "center":
            center = kD3 // 2
            w3[:, :, center, :, :].copy_(w2)
        else:
            raise ValueError(f"Unknown mode='{mode}', expected 'avg' or 'center'.")

        # Copy bias if present (identical)
        if b2 is not None and b3 is not None:
            b3.copy_(b2)


class FlexiMedDINOv3(MedDINOv3):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        # Call parent class constructor with all required parameters
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
            untie_cls_and_patch_norms=untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            device=device,
            **ignored_kwargs,
        )
        
        self.patch_embed_2D = PatchEmbedND(
            dim = 2,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embed_3D = PatchEmbedND(
            dim = 3,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

