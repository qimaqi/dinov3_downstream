from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
from torch import nn
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

import math

try:
    from natten import na2d
except ImportError as e:
    print("==== NATTEN is not installed. Please install it from the official project first. ==== ")
    na2d = None
# except ImportError as e:
#     raise ImportError(
#         "NATTEN is not installed. Please install it from the official project first."
#     ) from e


# class NATFeatureFusion(nn.Module):
#     """
#     DINO-guided NAT-based feature fusion.

#     Input:
#         dino_feat: [B, C, H, W]
#         cnn_feat : [B, C, H, W]

#     Output:
#         fused_feat: [B, C, H, W]

#     Design:
#         - Query from DINO
#         - Key/Value from CNN
#         - NAT produces local correction
#         - Residual add keeps DINO as the semantic backbone
#     """

#     def __init__(
#         self,
#         channels: int,
#         num_heads: int = 8,
#         kernel_size: int = 7,
#         dilation: int = 1,
#         alpha: float = 0.1,
#         use_gate: bool = True,
#         qkv_bias: bool = True,
#         proj_drop: float = 0.0,
#     ):
#         super().__init__()

#         if channels % num_heads != 0:
#             raise ValueError(
#                 f"channels ({channels}) must be divisible by num_heads ({num_heads})."
#             )

#         self.channels = channels
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.alpha = alpha
#         self.use_gate = use_gate

#         # Query from DINO
#         self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)

#         # Key/Value from CNN
#         self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
#         self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)

#         # Output projection for correction branch
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#         self.out_drop = nn.Dropout(proj_drop)

#         # Optional gate: decide where correction should be applied
#         if use_gate:
#             self.gate = nn.Sequential(
#                 nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
#                 nn.GELU(),
#                 nn.Conv2d(channels, 1, kernel_size=1, bias=True),
#                 nn.Sigmoid(),
#             )

#         # Optional norm-style scaling to stabilize
#         self.scale = self.head_dim ** -0.5

#     def _to_heads_last(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         [B, C, H, W] -> [B, H, W, heads, head_dim]
#         """
#         b, c, h, w = x.shape
#         x = x.view(b, self.num_heads, self.head_dim, h, w)
#         x = x.permute(0, 3, 4, 1, 2).contiguous()
#         return x

#     def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         [B, H, W, heads, head_dim] -> [B, C, H, W]
#         """
#         b, h, w, nh, hd = x.shape
#         x = x.permute(0, 3, 4, 1, 2).contiguous()
#         x = x.view(b, nh * hd, h, w)
#         return x

#     def forward(self, dino_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
#         if dino_feat.shape != cnn_feat.shape:
#             raise ValueError(
#                 f"dino_feat and cnn_feat must have the same shape, got "
#                 f"{dino_feat.shape} vs {cnn_feat.shape}"
#             )

#         # 1) projections
#         q = self.q_proj(dino_feat)   # [B, C, H, W]
#         k = self.k_proj(cnn_feat)    # [B, C, H, W]
#         v = self.v_proj(cnn_feat)    # [B, C, H, W]

#         # 2) convert to NATTEN format: [B, H, W, heads, head_dim]
#         q = self._to_heads_last(q)
#         k = self._to_heads_last(k)
#         v = self._to_heads_last(v)

#         # 3) NAT local cross-attention-like correction
#         # Query is from DINO, key/value from CNN.
#         delta = na2d(
#             q,
#             k,
#             v,
#             kernel_size=(self.kernel_size, self.kernel_size),
#             dilation=(self.dilation, self.dilation),
#             scale=self.scale,
#         )

#         # 4) back to [B, C, H, W]
#         delta = self._to_channels_first(delta)
#         delta = self.out_proj(delta)
#         delta = self.out_drop(delta)

#         # 5) constrain correction magnitude
#         delta = F.softsign(delta)
#         # torch.tanh(delta)

#         # 6) optional gate
#         if self.use_gate:
#             gate = self.gate(torch.cat([dino_feat, cnn_feat], dim=1))  # [B,1,H,W]
#             fused = dino_feat + self.alpha * gate * delta
#         else:
#             fused = dino_feat + self.alpha * delta

#         return fused

def normalize_no_norm_input(x, norm_type, in_chans=3):

    if norm_type is not None:
        if norm_type == 'meddinov3':
            # print("before norm", x.min(), x.max())
            x = torch.clamp(x, min=-1000, max=1000)
            mean = 65.1084213256836
            std = 178.01663208007812
            x = (x - mean) / (std + 1e-8)
            x = x.repeat(1, 3, 1, 1)

        elif norm_type == 'window':
            # x: B, 1, H, W  (torch tensor)

            wcenter = 50
            wwidth = 400
            wmin = wcenter - wwidth / 2.0
            wmax = wcenter + wwidth / 2.0

            x = torch.clamp(x, min=wmin, max=wmax)
            x = (x - wmin) / (wmax - wmin + 1e-6)

            # repeat 1 -> 3 channels
            x = x.repeat(1, 3, 1, 1)

            # DINO normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

            x = (x - mean) / std


        elif norm_type == 'multi_window':
            # x: B, 1, H, W  (torch tensor)

            windows = [
                (50, 400),    # soft tissue
                (40, 80),     # narrow soft tissue
                (600, 2800),  # bone
            ]

            x_multi = []

            for wcenter, wwidth in windows:
                wmin = wcenter - wwidth / 2.0
                wmax = wcenter + wwidth / 2.0

                x_w = torch.clamp(x, min=wmin, max=wmax)
                x_w = (x_w - wmin) / (wmax - wmin + 1e-6)

                x_multi.append(x_w)

            # concatenate -> B, 3, H, W
            x = torch.cat(x_multi, dim=1)

            # DINO normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

            x = (x - mean) / std

    elif in_chans==3:
        x = x.repeat(1, 3, 1, 1)
    elif in_chans==1:
        pass
    
    return x 


class NATFeatureFusion(nn.Module):
    """
    Pure PyTorch replacement for NATFeatureFusion.

    DINO-guided local cross-attention:
        - Query from DINO
        - Key/Value from CNN local neighborhoods
        - Residual correction added back to DINO

    Input:
        dino_feat: [B, C, H, W]
        cnn_feat : [B, C, H, W]

    Output:
        fused_feat: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        kernel_size: int = 5,
        dilation: int = 1,
        alpha: float = 0.1,
        use_gate: bool = True,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.alpha = alpha
        self.use_gate = use_gate

        # Query from DINO
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)

        # Key / Value from CNN
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_drop = nn.Dropout(proj_drop)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, dino_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        if dino_feat.shape != cnn_feat.shape:
            raise ValueError(
                f"dino_feat and cnn_feat must have the same shape, got "
                f"{dino_feat.shape} vs {cnn_feat.shape}"
            )

        b, c, h, w = dino_feat.shape
        # print("dino_feat.shape", dino_feat.shape)
        ksize = self.kernel_size
        pad = (ksize // 2) * self.dilation
        neighborhood_size = ksize * ksize

        # 1) Projections
        q = self.q_proj(dino_feat)   # [B, C, H, W]
        k = self.k_proj(cnn_feat)    # [B, C, H, W]
        v = self.v_proj(cnn_feat)    # [B, C, H, W]

        # 2) Reshape query: one query per spatial location
        # q -> [B, heads, H*W, 1, head_dim]
        q = q.view(b, self.num_heads, self.head_dim, h, w)
        q = q.permute(0, 1, 3, 4, 2).reshape(b, self.num_heads, h * w, 1, self.head_dim)

        # 3) Extract local K/V neighborhoods using unfold
        # unfold output: [B, C * K*K, H*W]
        k_unf = F.unfold(k, kernel_size=ksize, dilation=self.dilation, padding=pad, stride=1)
        v_unf = F.unfold(v, kernel_size=ksize, dilation=self.dilation, padding=pad, stride=1)

        # reshape to [B, heads, H*W, K*K, head_dim]
        k_unf = k_unf.view(b, self.num_heads, self.head_dim, neighborhood_size, h * w)
        k_unf = k_unf.permute(0, 1, 4, 3, 2).contiguous()

        v_unf = v_unf.view(b, self.num_heads, self.head_dim, neighborhood_size, h * w)
        v_unf = v_unf.permute(0, 1, 4, 3, 2).contiguous()

        # 4) Local cross-attention
        # q: [B, heads, L, 1, D]
        # k/v: [B, heads, L, K*K, D]
        # We merge [B, heads, L] into batch for SDPA
        q_flat = q.reshape(b * self.num_heads * h * w, 1, self.head_dim)
        k_flat = k_unf.reshape(b * self.num_heads * h * w, neighborhood_size, self.head_dim)
        v_flat = v_unf.reshape(b * self.num_heads * h * w, neighborhood_size, self.head_dim)

        delta = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # [B*heads*L, 1, D]

        # 5) Restore to [B, C, H, W]
        delta = delta.view(b, self.num_heads, h * w, self.head_dim)
        delta = delta.view(b, self.num_heads, h, w, self.head_dim)
        delta = delta.permute(0, 1, 4, 2, 3).contiguous().view(b, c, h, w)

        delta = self.out_proj(delta)
        delta = self.out_drop(delta)

        # 6) Constrain correction magnitude
        delta = F.softsign(delta)

        # 7) Residual correction
        if self.use_gate:
            gate = self.gate(torch.cat([dino_feat, cnn_feat], dim=1))  # [B,1,H,W]
            fused = dino_feat + self.alpha * gate * delta
        else:
            fused = dino_feat + self.alpha * delta

        return fused



class PatchDecodeTrilinear(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size


    def forward(self, x):
        # print("x shape", x.shape)
        if self.patch_size == 16:
            x1 = F.interpolate(
                x,
                scale_factor=(1, int(np.sqrt(self.patch_size)), int(np.sqrt(self.patch_size))),
                mode="trilinear",
                align_corners=False,
            )

            return F.interpolate(
                x1,
                scale_factor=(1, int(np.sqrt(self.patch_size)), int(np.sqrt(self.patch_size))),
                mode="trilinear",
                align_corners=False,
            )
        elif self.patch_size == 8:
            x1 = F.interpolate(
                x,
                scale_factor=(1, 2, 2),
                mode="trilinear",
                align_corners=False,
            )

            return F.interpolate(
                x1,
                scale_factor=(1, 4, 4),
                mode="trilinear",
                align_corners=False,
            )


class PatchDecodeBilinear(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False,
        )

class PrimusLinear3D(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=None,
        decoder_act=None,
        dino_encoder=None,
        freeze_backbone=True,
        target_shape_for_dino=None,
        free_tokenizer=False,
    ):
        super().__init__()


        # PatchDecodeBilinear(patch_embed_size)
        # PatchDecodeTrilinear(patch_embed_size)
        self.up_projection = PatchDecodeTrilinear(patch_embed_size)
        self.dino_encoder = dino_encoder
        self.free_tokenizer = free_tokenizer
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()
            if free_tokenizer:
                tokenizer = getattr(self.dino_encoder, "patch_embed", None)
                if tokenizer is None:
                    raise AttributeError("free_tokenizer=True requires dino_encoder.patch_embed to exist.")
                tokenizer.requires_grad_(True)
                tokenizer.train()

        self.decoder = Decoder()

        self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        self.target_shape_for_dino = target_shape_for_dino




    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # print("x_2d", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
            dino_features = self.dino_encoder.get_intermediate_layers(x_2d, n=1, reshape=True)[0]
        feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
        dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
        dino_features = dino_features.permute(0,2,1,3,4,)
        return dino_features


    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        assert len(x.shape) == 5
        # print("3D x", x.shape)
        # raise NotImplementedError('implement 3d version')

        x = x.repeat(1, 3,1, 1, 1)
        # print("X shape", x.shape)
        x = self.prepare_dino_features(x)

        # print("dino shape", x.shape )
        # b, c, d, h, w = x.shape
        # x = x.permute(0,2,1,3,4).reshape(b*d, c, h,w)
        
        dec_out = self.up_projection(x)
        # print("dec_out", dec_out.shape)

        # _,_, h_, w_ = dec_out.shape
        # dec_out = dec_out.reshape(b,d,c,h_,w_).permute(0,2,1,3,4)
        # print("dec_out", dec_out.shape)
        # dec_out torch.Size([8, 768, 640, 640])
        dec_out = self.seg_layer(dec_out)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PrimusLinear(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=None,
        decoder_act=None,
        dino_encoder=None,
        freeze_backbone=True,
        free_tokenizer=False,
        target_shape=None,
        norm_type=None,
        interaction_indice=None,
    ):
        super().__init__()

        self.up_projection = PatchDecodeBilinear(patch_embed_size)

        self.dino_encoder = dino_encoder
        self.free_tokenizer = free_tokenizer
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()
            if free_tokenizer:
                tokenizer = getattr(self.dino_encoder, "patch_embed", None)
                if tokenizer is None:
                    raise AttributeError("free_tokenizer=True requires dino_encoder.patch_embed to exist.")
                tokenizer.requires_grad_(True)
                tokenizer.train()

        self.decoder = Decoder()
        self.target_shape = target_shape
        self.norm_type = norm_type
        self.interaction_indice = interaction_indice

        self.seg_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        # if self.norm_type is not None:
        #     if self.norm_type == 'meddinov3':
        #         # print("before norm", x.min(), x.max())
        #         x = torch.clamp(x, min=-1000, max=1000)
        #         mean = 65.1084213256836
        #         std = 178.01663208007812
        #         x = (x - mean) / (std + 1e-8)
        #         x = x.repeat(1, 3, 1, 1)

        #     elif self.norm_type == 'window':
        #         # x: B, 1, H, W  (torch tensor)

        #         wcenter = 50
        #         wwidth = 400
        #         wmin = wcenter - wwidth / 2.0
        #         wmax = wcenter + wwidth / 2.0

        #         x = torch.clamp(x, min=wmin, max=wmax)
        #         x = (x - wmin) / (wmax - wmin + 1e-6)

        #         # repeat 1 -> 3 channels
        #         x = x.repeat(1, 3, 1, 1)

        #         # DINO normalization
        #         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        #         std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

        #         x = (x - mean) / std


        #     elif self.norm_type == 'multi_window':
        #         # x: B, 1, H, W  (torch tensor)

        #         windows = [
        #             (50, 400),    # soft tissue
        #             (40, 80),     # narrow soft tissue
        #             (600, 2800),  # bone
        #         ]

        #         x_multi = []

        #         for wcenter, wwidth in windows:
        #             wmin = wcenter - wwidth / 2.0
        #             wmax = wcenter + wwidth / 2.0

        #             x_w = torch.clamp(x, min=wmin, max=wmax)
        #             x_w = (x_w - wmin) / (wmax - wmin + 1e-6)

        #             x_multi.append(x_w)

        #         # concatenate -> B, 3, H, W
        #         x = torch.cat(x_multi, dim=1)

        #         # DINO normalization
        #         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        #         std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

        #         x = (x - mean) / std

        # else:
        #     x = x.repeat(1, 3, 1, 1)
        #         # print("after norm", x.min(), x.max())

        x = normalize_no_norm_input(
            x, norm_type=self.norm_type
        )

        
        if self.target_shape is not None:
            input_h, intput_w = x.shape[-2:]
            if intput_w != self.target_shape or input_h != self.target_shape:
                # print("Do interpolation")
                x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)

        if self.interaction_indice == None:
            x = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        else:
            assert len(self.interaction_indice) == 1, 'cross layer experiments'
            x = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indice , reshape=True)[0]

        # print("after forward", x.shape, 'x min', x.min(), 'x max', x.max())
        dec_out = self.up_projection(x)
        # after forward torch.Size([12, 768, 32, 32]) x min tensor(nan, device='cuda:0') x max tensor(nan, device='cuda:0')
        # print("dec_out", dec_out.shape)
        # dec_out torch.Size([8, 768, 640, 640])
        dec_out = self.seg_layer(dec_out)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")



class PrimusLinear_in_chans_1(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=None,
        decoder_act=None,
        dino_encoder=None,
        freeze_backbone=True,
        free_tokenizer=False,
        target_shape=None,
        norm_type=None,
        interaction_indice=None,
    ):
        super().__init__()

        self.up_projection = PatchDecodeBilinear(patch_embed_size)

        self.dino_encoder = dino_encoder
        self.free_tokenizer = free_tokenizer
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()
            if free_tokenizer:
                tokenizer = getattr(self.dino_encoder, "patch_embed", None)
                if tokenizer is None:
                    raise AttributeError("free_tokenizer=True requires dino_encoder.patch_embed to exist.")
                tokenizer.requires_grad_(True)
                tokenizer.train()

        self.decoder = Decoder()
        self.target_shape = target_shape
        self.norm_type = norm_type
        self.interaction_indice = interaction_indice

        self.seg_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = normalize_no_norm_input(
            x, norm_type=self.norm_type, in_chans=1
        )

        
        if self.target_shape is not None:
            input_h, intput_w = x.shape[-2:]
            if intput_w != self.target_shape or input_h != self.target_shape:
                # print("Do interpolation")
                x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)

        if self.interaction_indice == None:
            x = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        else:
            assert len(self.interaction_indice) == 1, 'cross layer experiments'
            x = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indice , reshape=True)[0]


        dec_out = self.up_projection(x)
        # print("dec_out", dec_out.shape, 'dec_out min', dec_out.min(), 'dec_out max', dec_out.max())
        # dec_out torch.Size([8, 768, 640, 640])
        dec_out = self.seg_layer(dec_out)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")



class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int, 
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2 ** n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1]//2)
        ch.append(out_channels)
        
        self.ch = ch
        print("PatchDecoder dim", ch, 'n', n)

        stages = []
        for i in range(n):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                    norm(ch[i + 1]),
                    activation(),
                )
            )
        if out_channels != 0:
            stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)



class PatchDecode2_5D(nn.Module):
    """
    Stable decoder for shallow 3D feature:
    [B, C, D, H, W]
    where D is shallow (1~16), H/W large.

    Strategy:
    - early stages: only XY upsample
    - feature channels //2 each stage
    - light XY smoothing
    - final Z-aware refinement
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        super().__init__()

        assert len(patch_size) == 3
        _, py, px = patch_size

        n = int(math.log2(max(py, px)))
        assert 2 ** n == max(py, px)

        # channel schedule
        ch = [embed_dim]
        for _ in range(n):
            ch.append(max(ch[-1] // 2, out_channels))

        self.ch = ch
        print("PatchDecode2_5D channels:", ch)

        stages = []


        for i in range(n):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        ch[i],
                        ch[i + 1],
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2),
                    ),
                    norm(ch[i + 1]),
                    activation(),
                )
            )

        # ---------------------------------------------------
        # final shallow Z refinement
        # ---------------------------------------------------
        # stages.append(
        #     nn.Sequential(
        #         nn.Conv3d(
        #             ch[-1],
        #             ch[-1],
        #             kernel_size=(3, 1, 1),
        #             padding=(1, 0, 0),
        #         ),
        #         norm(ch[-1]),
        #         activation(),
        #     )
        # )

        # ---------------------------------------------------
        # output projection
        # ---------------------------------------------------
        stages.append(
            nn.Conv3d(ch[-1], out_channels, kernel_size=1)
        )

        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        return self.decode(x)


class PatchDecode3D(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        print("strides in PatchDecode3D", strides)
        if out_channels != 0:
            dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        else:
            dim_red = (embed_dim / (2 * 96)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        self.channels = channels
        print("channels in PatchDecode3D", self.channels, 'num_stages', num_stages)
        
        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                    norm(channels[s + 1]),
                    activation(),
                )
            )

        if out_channels != 0:
            # stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
            stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        else:
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[-2], 192,  kernel_size=strides[-1], stride=strides[-1]),
                    norm(192),
                    activation(),
                )
            )

        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """

        return self.decode(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        norm_type=None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )
        self.norm_type = norm_type

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1

        x = normalize_no_norm_input(
            x, norm_type=self.norm_type
        )

        # x = x.repeat(1,3,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class Primus3D(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode3D(
            (1, patch_embed_size, patch_embed_size), embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        # if freeze_backbone:
        #     self.dino_encoder.requires_grad_(False)
        #     self.dino_encoder.eval()
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True   
                    print("Name require grad", name)   


            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino =target_shape_for_dino


    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # print("x_2d shape", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
            dino_features = self.dino_encoder.get_intermediate_layers(x_2d, n=1, reshape=True)[0]
        feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
        dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
        dino_features = dino_features.permute(0,2,1,3,4,)
        return dino_features


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)

        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3,1, 1, 1)
        x = self.prepare_dino_features(x)
        
        dec_out = self.up_projection(x)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class Primus3D_cycle(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode3D(
            (1, patch_embed_size, patch_embed_size), embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino = target_shape_for_dino


    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        # x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # # print("x_2d shape", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
                x = x_2d.reshape(b, d, c, self.target_shape_for_dino, self.target_shape_for_dino).permute(0,2,1,3,4)
            dino_features = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        # feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
        # dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
        # dino_features = dino_features.permute(0,2,1,3,4,)
        return dino_features


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)

        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3, 1, 1, 1)
        x = self.prepare_dino_features(x)
        
        dec_out = self.up_projection(x)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class Primus3D_Multiscale_cycle(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
        interaction_indices =[2,5,8,11]
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode3D(
            (1, patch_embed_size, patch_embed_size), embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino = target_shape_for_dino
        self.interaction_indices = interaction_indices


    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        # x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # # print("x_2d shape", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
                x = x_2d.reshape(b, d, c, self.target_shape_for_dino, self.target_shape_for_dino).permute(0,2,1,3,4)
                
            dino_features = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
            dino_features = torch.cat(dino_features, dim=1)

        # feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
        # dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
        # dino_features = dino_features.permute(0,2,1,3,4,)
        return dino_features


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)

        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3,1, 1, 1)
        x = self.prepare_dino_features(x)
        
        dec_out = self.up_projection(x)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class Primus3D_linear_cycle(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecodeTrilinear(
            patch_embed_size,
        )
        self.target_shape_for_dino = target_shape_for_dino


        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        # self.up_projection.apply(InitWeights_He(1e-2))
        self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)


    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        # x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # # print("x_2d shape", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
                x = x_2d.reshape(b, d, c, self.target_shape_for_dino, self.target_shape_for_dino).permute(0,2,1,3,4)
            dino_features = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        # feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
        # dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
        # dino_features = dino_features.permute(0,2,1,3,4,)
        return dino_features


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)

        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3,1, 1, 1)
        x = self.prepare_dino_features(x)
        
        dec_out = self.up_projection(x)

        return self.seg_layer(dec_out)

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    

class Primus_Multiscale(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True   
                    print("Name require grad", name)   


            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()
                    
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1, f"{x.shape} is not supported"
        # x = x.repeat(1,3,1,1)
        assert x.shape[1] <= 3,f"{x.shape} is not supported"
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, 1:2, :, :]], dim=1)

        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")




class Primus_Multiscale_input_x_ch(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=False,
        interaction_indices=[1, 2, 3, 4],
        expected_input_channels=None,
    ):
        """Multiscale Primus variant that consumes the input channels as-is."""
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True
                    print("Name require grad", name)

            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices = interaction_indices
        self.expected_input_channels = expected_input_channels

    def forward(self, x, ret_mask=False):
        if self.expected_input_channels is not None:
            assert x.shape[1] == self.expected_input_channels, (
                f"Expected {self.expected_input_channels} input channels but got {x.shape[1]}"
            )
        assert x.shape[1] >= 1, f"{x.shape} is not supported"

        hier = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")

class DPT_Multiscale(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        interaction_indices =[1,2,3,4],
        features=128,
        use_bn=False,
        out_channels=[96, 192, 384, 768], 
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.dpt import DPTHead 
        # self.up_projection = PatchDecode(
        #     patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        # )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.head = DPTHead(num_classes, embed_dim, features, use_bn, out_channels=out_channels)
        

        # self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        # hier = torch.cat(hier, dim=1)
        # dec_out = self.up_projection(hier)
        out = self.head(hier, patch_h, patch_w)
        dec_out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")



class Primus_Multiscale3D(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
        interaction_indices =[1,2,3,4]
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection =  nn.Sequential(
                PatchDecode3D(
                    (1, patch_embed_size, patch_embed_size),
                    embed_dim * 4,
                    96,
                    norm=decoder_norm,
                    activation=decoder_act,
                ),
                nn.Conv3d(96, num_classes, kernel_size=1),
            )
        self.interaction_indices = interaction_indices

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True   
                    print("Name require grad", name)   

            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()


        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino =target_shape_for_dino


    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w) # (b*d, c, h, w)
        # print("x_2d shape", x_2d.shape)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
            dino_features_list = self.dino_encoder.get_intermediate_layers(x_2d, n=self.interaction_indices, reshape=True)
            dino_features_list_new = []

            for dino_features_i in dino_features_list:
                feature_c, feature_h, feature_w = dino_features_i.shape[1], dino_features_i.shape[2], dino_features_i.shape[3]
                dino_features_i = dino_features_i.reshape(b,d, feature_c,feature_h,feature_w)
                dino_features_i = dino_features_i.permute(0,2,1,3,4,)
                
                dino_features_list_new.append(dino_features_i)

        hier = torch.cat(dino_features_list_new, dim=1)
        # print("hier", hier.shape)
        # dec_out = self.up_projection(hier)
        return hier


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)

        # assert x.shape[1] == 1
        assert x.shape[1] <= 3,f"{x.shape} is not supported"
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1,1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, 1:2, :, :]], dim=1)


        assert len(x.shape) == 5
        # x = x.repeat(1,3,1,1,1)
        B,C,D,H,W = x.shape
        x_2d = x.permute(0,2,1,3,4).contiguous()
        x_2d = x_2d.reshape(B*D, C, H, W)
        # print("x_2d", x_2d.shape)
        if self.target_shape_for_dino is not None:
            x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)

        hier = self.dino_encoder.get_intermediate_layers(x_2d,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        B_,C_,H_,W_ = hier.shape
        hier = hier.reshape(B, D, C_, H_, W_).contiguous()
        hier = hier.permute(0,2,1,3,4)
        
        dec_out = self.up_projection(hier)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    

class Primus_Multiscale2D_debug(Primus_Multiscale):

    def forward(self, x, ret_mask=False):
        # now it is 3D input
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1,1)
        B,C,D,H,W = x.shape
        x_2d = x.permute(0,2,1,3,4)
        x_2d = x_2d.reshape(B*D, C, H, W)
        # print("x_2d", x_2d.shape)
        hier = self.dino_encoder.get_intermediate_layers(x_2d,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        B_,C_,H_,W_ = dec_out.shape
        dec_out = dec_out.reshape(B, D, C_, H_, W_)
        dec_out = dec_out.permute(0,2,1,3,4)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class Primus_Multiscale2_5D(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
        interaction_indices =[1,2,3,4]
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        # self.up_projection = PatchDecode(
        #     patch_embed_size, embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
        # )
        self.up_projection =  PatchDecode2_5D(
                    (1, patch_embed_size, patch_embed_size),
                    embed_dim * 4,
                    num_classes,
                    norm=decoder_norm,
                    activation=decoder_act,
                )
            
        self.interaction_indices = interaction_indices

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True   
                    print("Name require grad", name)   

            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()


        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino =target_shape_for_dino

    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
     
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b*d, c, h, w) # (b*d, c, h, w)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
                
            dino_features_list = self.dino_encoder.get_intermediate_layers(x_2d, n=self.interaction_indices, reshape=True)
            # dino_features_list_new = []

            hier = torch.cat(dino_features_list, dim=1)
            # print("hier", hier.shape)
            feature_c, feature_h, feature_w = hier.shape[1], hier.shape[2], hier.shape[3]
            hier = hier.reshape(b,d, feature_c,feature_h,feature_w)
            hier = hier.permute(0,2,1,3,4).contiguous()

        return hier


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)
        x = x.repeat(1,3,1,1,1)
        B,C,D,H,W = x.shape
        x_2d = x.permute(0,2,1,3,4).contiguous()
        x_2d = x_2d.reshape(B*D, C, H, W)
        # print("x_2d", x_2d.shape)
        hier = self.dino_encoder.get_intermediate_layers(x_2d,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        B_,C_,H_,W_ = hier.shape
        hier = hier.reshape(B, D, C_, H_, W_).contiguous()
        hier = hier.permute(0,2,1,3,4)


        # assert x.shape[1] == 1
        # assert len(x.shape) == 5
        # x = x.repeat(1, 3, 1, 1, 1)
        # x = self.prepare_dino_features(x)

        # b, c, d, h, w = x.shape
        # x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b*d, c, h, w) # (b*d, c, h, w)
        dec_out = self.up_projection(hier)
        # B_,C_,H_,W_ = dec_out.shape
        # dec_out = dec_out.reshape(B, D, C_, H_, W_)
        # dec_out = dec_out.permute(0,2,1,3,4)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class Primus_Multiscale2_5D_debug(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        target_shape_for_dino=None,
        interaction_indices =[1,2,3,4]
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
        )
        # self.up_projection =  PatchDecode2_5D(
        #             (1, patch_embed_size, patch_embed_size),
        #             embed_dim * 4,
        #             num_classes,
        #             norm=decoder_norm,
        #             activation=decoder_act,
        #         )
            
        self.interaction_indices = interaction_indices

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            for name, p in self.dino_encoder.named_parameters():
                if "Inner_Adapter" in name:
                    p.requires_grad = True   
                    print("Name require grad", name)   

            self.dino_encoder.eval()

            for name, m in self.dino_encoder.named_modules():
                if "Inner_Adapter" in name:
                    print("Name for .train()", name)
                    m.train()


        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.target_shape_for_dino =target_shape_for_dino

    def prepare_dino_features(self, x):
        """
        x: (b, c, d, h, w)
        """
        assert x.shape[1] == 3, 'need to be preprocessed'
        b, c, d, h, w = x.shape
     
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b*d, c, h, w) # (b*d, c, h, w)
        with torch.no_grad():
            if self.target_shape_for_dino is not None:
                x_2d = nn.functional.interpolate(x_2d, size=(self.target_shape_for_dino, self.target_shape_for_dino), mode='bilinear', align_corners=False)
                
            dino_features_list = self.dino_encoder.get_intermediate_layers(x_2d, n=self.interaction_indices, reshape=True)
            # dino_features_list_new = []

            hier = torch.cat(dino_features_list, dim=1)
            # print("hier", hier.shape)
            feature_c, feature_h, feature_w = hier.shape[1], hier.shape[2], hier.shape[3]
            hier = hier.reshape(b,d, feature_c,feature_h,feature_w)
            hier = hier.permute(0,2,1,3,4).contiguous()

        return hier


    def forward(self, x, ret_mask=False):
        # assert x.shape[1] == 1
        # x = x.repeat(1,3,1,1)
        # x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        # dec_out = self.up_projection(x)
        x = x.repeat(1,3,1,1,1)
        B,C,D,H,W = x.shape
        x_2d = x.permute(0,2,1,3,4).contiguous()
        x_2d = x_2d.reshape(B*D, C, H, W)
        # print("x_2d", x_2d.shape)
        hier = self.dino_encoder.get_intermediate_layers(x_2d,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        # B_,C_,H_,W_ = hier.shape
        # hier = hier.reshape(B, D, C_, H_, W_)
        # hier = hier.permute(0,2,1,3,4)


        # assert x.shape[1] == 1
        # assert len(x.shape) == 5
        # x = x.repeat(1, 3, 1, 1, 1)
        # x = self.prepare_dino_features(x)

        # b, c, d, h, w = x.shape
        # x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b*d, c, h, w) # (b*d, c, h, w)
        dec_out = self.up_projection(hier)
        B_,C_,H_,W_ = dec_out.shape
        dec_out = dec_out.reshape(B, D, C_, H_, W_)
        dec_out = dec_out.permute(0,2,1,3,4)

        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    


class MultiScale(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    



# class PrimusLinearUNetAdapter(AbstractDynamicNetworkArchitectures):

#     def __init__(
#         self,
#         embed_dim: int,
#         patch_embed_size: int,
#         num_classes: int,
#         decoder_norm=None,
#         decoder_act=None,
#         dino_encoder=None,
#         freeze_backbone=True,
#         target_shape=512,
#         unet_adapter_type='m',
#         feature_extraction=False,
#         linear_prob=False,
#         image_only=False,
#     ):
#         super().__init__()

#         # self.up_projection = PatchDecodeBilinear(patch_embed_size)

#         self.dino_encoder = dino_encoder
#         self.feature_extraction =feature_extraction
#         if freeze_backbone:
#             self.dino_encoder.requires_grad_(False)
#             self.dino_encoder.eval()



#         self.decoder = Decoder()
#         self.target_shape = target_shape
#         self.image_only = image_only
#         self.embed_dim = embed_dim

#         self.dino_feature_fusion_layers = []
#         if unet_adapter_type == 'm':
#             self.adapter_features_per_stage = [32, 64, 128, 256, 384]
#             self.fusion_features_per_stage = [96,160,256,384,512]
#             # back to first 

#             self.strides_list = [[1,1],[2,2],[2,2],[2,2],[2,2]]
#             self.unet_adapter = ResidualEncoder(
#                 input_channels=1,
#                 n_stages=5,
#                 features_per_stage=self.adapter_features_per_stage,
#                 strides = self.strides_list,
#                 conv_op=torch.nn.modules.conv.Conv2d,
#                 kernel_sizes=[
#                     [3,3],
#                     [3,3],
#                     [3,3],
#                     [3,3],
#                     [3,3]
#                 ],
#                 n_blocks_per_stage=[1,3,4,6,6],  
#                 return_skips=True,
#                 norm_op=torch.nn.modules.instancenorm.InstanceNorm2d,
#                 norm_op_kwargs={
#                         "eps": 1e-05,
#                         "affine": True
#                     },
#                 nonlin = torch.nn.LeakyReLU,
#                 nonlin_kwargs = {
#                         "inplace": True
#                     }
#             )
#             output_feature_channels = self.fusion_features_per_stage[0]

#             for i in range(len(self.adapter_features_per_stage)):
   
#                     # self.dino_feature_fusion_layers.append(torch.nn.modules.conv.Conv2d(self.features_per_stage[i] + embed_dim, self.features_per_stage[i], 3, 1, 1, bias=True))
#                 self.dino_feature_fusion_layers.append(
#                     StackedResidualBlocks(
#                         1, torch.nn.modules.conv.Conv2d, self.adapter_features_per_stage[i] + embed_dim if i==len(self.adapter_features_per_stage)-1 else self.adapter_features_per_stage[i] + self.fusion_features_per_stage[i+1], self.fusion_features_per_stage[i], 3, 1,
#                         True, torch.nn.modules.instancenorm.InstanceNorm2d, {
#                         "eps": 1e-05,
#                         "affine": True
#                         }, 
#                         nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True}
#                     )
#                 )
#                 # else:
#                 #     self.dino_feature_fusion_layers.append(torch.nn.modules.conv.Conv2d(self.features_per_stage[i]*2, self.features_per_stage[i], 3, 1, 1,bias=True))

#             self.dino_feature_fusion_layers =  nn.ModuleList(self.dino_feature_fusion_layers)
#             # prepare linear mapping
            

#         else:
#             raise NotImplementedError('debug')

#         if linear_prob:
#             # self.unet_adapter
#             # self.dino_feature_fusion_layers
#             self.unet_adapter.requires_grad_(False)
#             self.unet_adapter.eval()      

#             self.dino_feature_fusion_layers.requires_grad_(False)
#             self.dino_feature_fusion_layers.eval()      


#         if not feature_extraction:
#             self.seg_layer = nn.Conv2d(output_feature_channels, num_classes, kernel_size=1)

#     def forward(self, x, ret_mask=False):
#         assert x.shape[1] == 1
#         skips = self.unet_adapter(x)
#         # for i, skip_i in enumerate(skips):
#         #     print("skip", i ,"size", skip_i.shape)
#         # skip 0 size torch.Size([12, 32, 512, 512])
#         # skip 1 size torch.Size([12, 64, 256, 256])
#         # skip 2 size torch.Size([12, 128, 128, 128])
#         # skip 3 size torch.Size([12, 256, 64, 64])
#         # skip 4 size torch.Size([12, 512, 32, 32])

#         x = x.repeat(1, 3, 1, 1)
#         # print("input x", x.shape)
#         # input x torch.Size([8, 3, 640, 640])
#         input_h = x.shape[-2]
#         input_w = x.shape[-1]
#         if not self.image_only:
#             if input_w != self.target_shape or input_h != self.target_shape:
#                 # print("Do interpolation")
#                 x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
#             x = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
#         else:
#             x = torch.zeros(x.shape[0], self.embed_dim,  self.target_shape // 16, self.target_shape // 16).to(x.device)

#         for s in range(len(self.adapter_features_per_stage )):
#             # print("x", x.shape, 'skips[-(s+1)]', skips[-(s+1)].shape)
#             x = torch.cat([x, skips[-(s+1)]], dim=1)
#             x = self.dino_feature_fusion_layers[-(s+1)](x)
#             # upsample here
#             x_interpolate = torch.nn.functional.interpolate(
#                 input=x,
#                 scale_factor=self.strides_list[-(s+1)],
#                 mode='bilinear',
#                 align_corners=False
#             )

#             x = x_interpolate

#         if not self.feature_extraction:
#             dec_out = self.seg_layer(x)
#         return dec_out

#     def compute_conv_feature_map_size(self, input_size):
#         raise NotImplementedError("yuck")




class PrimusLinearUNetAdapterNaT(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=None,
        decoder_act=None,
        dino_encoder=None,
        freeze_backbone=True,
        target_shape=512,
        unet_adapter_type='m',
        feature_extraction=False,
        linear_prob=False,
        image_only=False,
        interaction_indices=[2,5,8,11]

    ):
        super().__init__()

        # self.up_projection = PatchDecodeBilinear(patch_embed_size)

        self.dino_encoder = dino_encoder
        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()



        self.decoder = Decoder()
        self.target_shape = target_shape
        self.image_only = image_only
        self.embed_dim = embed_dim
        self.interaction_indices=interaction_indices

        # self.dino_feature_fusion_layers = []
        # self.feature_dimension_compression_layers = []
        if unet_adapter_type == 'm':
            self.adapter_features_per_stage = [32, 64, 128, 256, 512]
            self.compression_embedding_dim = 512
            self.merge_layer_from_nnunet = 4

            self.strides_list = [[1,1],[2,2],[2,2],[2,2],[2,2]]
            self.unet_adapter = ResidualEncoder(
                input_channels=1,
                n_stages=5,
                features_per_stage=self.adapter_features_per_stage,
                strides = self.strides_list,
                conv_op=torch.nn.modules.conv.Conv2d,
                kernel_sizes=[
                    [3,3],
                    [3,3],
                    [3,3],
                    [3,3],
                    [3,3]
                ],
                n_blocks_per_stage=[1,3,4,6,6],  
                return_skips=True,
                norm_op=torch.nn.modules.instancenorm.InstanceNorm2d,
                norm_op_kwargs={
                        "eps": 1e-05,
                        "affine": True
                    },
                nonlin = torch.nn.LeakyReLU,
                nonlin_kwargs = {
                        "inplace": True
                    }
            )
            output_feature_channels = self.compression_embedding_dim

            self.dino_feature_fusion_layers = NATFeatureFusion(
                channels=self.compression_embedding_dim,
                num_heads=8,
                kernel_size=7,
                dilation=1,
                alpha=0.1,
                use_gate=True,
            )

            self.feature_dimension_compression_layers = StackedResidualBlocks(
                        2, torch.nn.modules.conv.Conv2d, embed_dim*4, self.compression_embedding_dim, 3, 1,
                        True, torch.nn.modules.instancenorm.InstanceNorm2d, {
                        "eps": 1e-05,
                        "affine": True
                        }, 
                        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True}
                    )

            self.up_projection = PatchDecodeBilinear(
                patch_embed_size
            )

        else:
            raise NotImplementedError('debug')

        if linear_prob:
            self.unet_adapter.requires_grad_(False)
            self.unet_adapter.eval()      

            self.dino_feature_fusion_layers.requires_grad_(False)
            self.dino_feature_fusion_layers.eval()      


        if not feature_extraction:
            self.seg_layer = nn.Conv2d(output_feature_channels, num_classes, kernel_size=1)

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        skips = self.unet_adapter(x)
        # for i, skip_i in enumerate(skips):
        #     print("skip", i ,"size", skip_i.shape)
        # skip 0 size torch.Size([12, 32, 512, 512])
        # skip 1 size torch.Size([12, 64, 256, 256])
        # skip 2 size torch.Size([12, 128, 128, 128])
        # skip 3 size torch.Size([12, 256, 64, 64])
        # skip 4 size torch.Size([12, 512, 32, 32])

        x = x.repeat(1, 3, 1, 1)
        # print("input x", x.shape)
        # input x torch.Size([8, 3, 640, 640])
        input_h = x.shape[-2]
        input_w = x.shape[-1]
        if not self.image_only:
            if input_w != self.target_shape or input_h != self.target_shape:
                x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
            x = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)
        else:
            raise NotImplementedError('To dev')
            # x = torch.zeros(x.shape[0], self.embed_dim,  self.target_shape // 16, self.target_shape // 16).to(x.device)

        # print("x", len(x), 'self.interaction_indices', self.interaction_indices)
        x = torch.cat(x, dim=1)
        x = self.feature_dimension_compression_layers(x)

        x = self.dino_feature_fusion_layers(x, skips[-1])

        x = self.up_projection(x)

        # for s in range(len(self.adapter_features_per_stage )):
        #     # print("x", x.shape, 'skips[-(s+1)]', skips[-(s+1)].shape)
        #     x = torch.cat([x, skips[-(s+1)]], dim=1)
        #     x = self.dino_feature_fusion_layers[-(s+1)](x)
        #     # upsample here
        #     x_interpolate = torch.nn.functional.interpolate(
        #         input=x,
        #         scale_factor=self.strides_list[-(s+1)],
        #         mode='bilinear',
        #         align_corners=False
        #     )

        #     x = x_interpolate

        if not self.feature_extraction:
            dec_out = self.seg_layer(x)
        else:
            dec_out = x
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
