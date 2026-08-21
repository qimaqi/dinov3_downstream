from typing import Tuple

import torch
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from dinov3.models.primus import Decoder, PatchDecode3D, PatchDecodeTrilinear


class _SliceWiseDino3DBase(AbstractDynamicNetworkArchitectures):
    """Base wrapper for DINO encoders that already support 5D slice-wise feature extraction."""

    def __init__(
        self,
        *,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        dino_encoder: nn.Module,
        freeze_backbone: bool = True,
        target_shape_for_dino: int | Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_size = patch_embed_size
        self.num_classes = num_classes
        self.dino_encoder = dino_encoder
        self.target_shape_for_dino = target_shape_for_dino
        self.decoder = Decoder()

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

    def _repeat_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, D, H, W), got {tuple(x.shape)}")
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, 1:2, :, :, :]], dim=1)

        if x.shape[1] == 3:
            return x
        raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")

    def _resize_volume_slices(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_shape_for_dino is None:
            return x
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        x_2d = nn.functional.interpolate(
            x_2d,
            size=self.target_shape_for_dino,
            mode="bilinear",
            align_corners=False,
        )
        h_new, w_new = x_2d.shape[-2:]
        return x_2d.reshape(b, d, c, h_new, w_new).permute(0, 2, 1, 3, 4).contiguous()

    def prepare_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        # print("input x", x.shape)
        x = self._repeat_to_rgb(x)
        x = self._resize_volume_slices(x)
        # print("before vit x", x.shape)
        grad_enabled = any(p.requires_grad for p in self.dino_encoder.parameters())
        with torch.set_grad_enabled(grad_enabled):
            features = self.dino_encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        if features.ndim != 5:
            raise ValueError(
                "dino_encoder.get_intermediate_layers(..., reshape=True) must return "
                f"a 5D feature map for slice-wise 3D decoding, got {tuple(features.shape)}"
            )
        return features

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("Feature map size estimation is not implemented for slice-wise DINO decoders.")


class Linear3D(_SliceWiseDino3DBase):
    """Linear 3D decoder fed by 5D slice-wise DINO features."""

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=None,
        decoder_act=None,
        dino_encoder=None,
        freeze_backbone: bool = True,
        target_shape_for_dino: int | Tuple[int, int] | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            num_classes=num_classes,
            dino_encoder=dino_encoder,
            freeze_backbone=freeze_backbone,
            target_shape_for_dino=target_shape_for_dino,
        )
        self.up_projection = PatchDecodeTrilinear(patch_embed_size)
        self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, ret_mask: bool = False) -> torch.Tensor:
        features = self.prepare_dino_features(x)
        dec_out = self.up_projection(features)
        return self.seg_layer(dec_out)


class Decoder3D(_SliceWiseDino3DBase):
    """PatchDecode3D decoder fed by 5D slice-wise DINO features."""

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone: bool = True,
        target_shape_for_dino: int | Tuple[int, int] | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            num_classes=num_classes,
            dino_encoder=dino_encoder,
            freeze_backbone=freeze_backbone,
            target_shape_for_dino=target_shape_for_dino,
        )
        self.up_projection = PatchDecode3D(
            (1, patch_embed_size, patch_embed_size),
            embed_dim,
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x: torch.Tensor, ret_mask: bool = False) -> torch.Tensor:
        features = self.prepare_dino_features(x)
        return self.up_projection(features)
