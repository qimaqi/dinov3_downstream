"""Flexi-CT segmentation heads used by the downstream nnU-Net trainer."""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from .flexi_ct_2d import Flexi_CT_2D, _BACKBONE_KWARGS, _load_teacher_into_backbone
from .flexi_ct_3d import Flexi_CT_3D
from .models import flexi_ct_backbone_base


_DEFAULT_INTERACTION_INDICES = (3, 7, 11, 15)
_DEFAULT_SINGLE_INTERACTION_INDICE = 15
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
import numpy as np

import math

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
        dim=2
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

        stages = []

        if dim == 2:
            for i in range(n):
                stages.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                        norm(ch[i + 1]),
                        activation(),
                    )
                )
            stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        elif dim == 3:
            for i in range(n):
                stages.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                        norm(ch[i + 1]),
                        activation(),
                    )
                )
            stages.append(nn.Conv3d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class LinearBiUpsampler(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        patch_size: int | Sequence[int],
        dim: int = 2,
    ):
        super().__init__()
        self.dim = dim
        if isinstance(patch_size, Sequence) and not isinstance(patch_size, (str, bytes)):
            self.scale_factor = tuple(patch_size)
        else:
            self.scale_factor = patch_size

        if dim == 2:
            self.classifier = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=True)
            self.interpolation_mode = "bilinear"
        elif dim == 3:
            self.classifier = nn.Conv3d(embed_dim, out_channels, kernel_size=1, bias=True)
            self.interpolation_mode = "trilinear"
        else:
            raise ValueError(f"Unsupported dim={dim}, expected 2 or 3")

    def forward(self, x, target_size=None):
        if target_size is None:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.interpolation_mode,
                align_corners=False,
            )
        else:
            x = F.interpolate(
                x,
                size=target_size,
                mode=self.interpolation_mode,
                align_corners=False,
            )
        return self.classifier(x)


class Primus(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.in_chans = in_chans
        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act, dim=dim
        )
        self.dim = dim
        self.decoder.apply(InitWeights_He(1e-2))

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.decoder(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    
class Primus_v2(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.dim = dim
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act, 
            dim = dim
        )
        proj_dim = (embed_dim * len(interaction_indices))
        
        if dim == 2:
            self.projectors =  nn.Sequential(
                    nn.Conv2d(
                        proj_dim,
                        embed_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNormNd(embed_dim),
                    )
        else:
            self.projectors =  nn.Sequential(
                    nn.Conv3d(
                        proj_dim,
                        embed_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNormNd(embed_dim),
                    )
        
        self.in_chans = in_chans
        self.dino_encoder = dino_encoder
        self.decoder.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        hier = self.projectors(hier)
        dec_out = self.decoder(hier)
        return dec_out
    
class Primus_Onescale(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        dim=2,
        interaction_indices: int | Sequence[int] = 1,
        decoder_type: Literal["patch", "linear"] = "patch",
        use_projector: bool = False,
    ):
        """
        One-scale Primus head that consumes a single intermediate layer.
        The feature can optionally be projected first, then sent through either PatchDecode or a
        linear-probe style upsampler that resizes directly to the target resolution before a
        single linear classifier.
        """
        super().__init__()

        self.dim = dim
        self.in_chans = in_chans
        self.dino_encoder = dino_encoder
        self.decoder_type = decoder_type
        self.use_projector = use_projector
        self.project_dim = embed_dim // 2

        if isinstance(interaction_indices, Sequence) and not isinstance(interaction_indices, (str, bytes)):
            if len(interaction_indices) != 1:
                raise ValueError(
                    "Primus_Onescale expects exactly one interaction index; "
                    f"got {len(interaction_indices)} indices."
                )
            interaction_index = interaction_indices[0]
        else:
            interaction_index = interaction_indices
        self.interaction_indices = interaction_index

        decoder_in_dim = self.project_dim if use_projector else embed_dim

        if use_projector:
            proj_layers = []
            if dim == 2:
                proj_layers.extend(
                    [
                        nn.Conv2d(embed_dim, self.project_dim, kernel_size=1, bias=False),
                        LayerNormNd(self.project_dim),
                    ]
                )
            elif dim == 3:
                proj_layers.extend(
                    [
                        nn.Conv3d(embed_dim, self.project_dim, kernel_size=1, bias=False),
                        LayerNormNd(self.project_dim),
                    ]
                )
            else:
                raise ValueError(f"Unsupported dim={dim}, expected 2 or 3")
            self.projectors = nn.Sequential(*proj_layers)
        else:
            self.projectors = nn.Identity()

        if decoder_type == "patch":
            self.decoder = PatchDecode(
                patch_embed_size,
                decoder_in_dim,
                num_classes,
                norm=decoder_norm,
                activation=decoder_act,
                dim=dim,
            )
        elif decoder_type == "linear":
            self.decoder = LinearBiUpsampler(
                embed_dim=decoder_in_dim,
                out_channels=num_classes,
                patch_size=patch_embed_size,
                dim=dim,
            )
        else:
            raise ValueError(f"Unsupported decoder_type={decoder_type!r}, expected 'patch' or 'linear'")

        self.decoder.apply(InitWeights_He(1e-2))

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1, self.in_chans, 1, 1)
            elif x.dim() == 5:
                x = x.repeat(1, self.in_chans, 1, 1, 1)
        hier = self.dino_encoder.get_intermediate_layers(x, n=[self.interaction_indices], reshape=True)[0]
        hier = self.projectors(hier)
        if self.decoder_type == "linear":
            dec_out = self.decoder(hier, target_size=x.shape[2:])
        else:
            dec_out = self.decoder(hier)
        return dec_out


class Primus_Multiscale(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.dim = dim
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act, 
            dim = dim
        )
        self.in_chans = in_chans
        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.decoder(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
