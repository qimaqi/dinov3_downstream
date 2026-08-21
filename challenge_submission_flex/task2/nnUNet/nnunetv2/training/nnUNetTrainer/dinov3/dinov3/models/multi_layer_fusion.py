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
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd

class ChannelGate3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(x)

class AlignBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=32, use_gate=True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

        self.use_gate = use_gate
        if use_gate:
            self.gate = ChannelGate(out_ch)

        self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.res_proj(x)
        out = self.proj(x)
        out = self.norm(out)
        out = self.act(out)

        if self.use_gate:
            out = self.gate(out)

        out = out + identity
        return out


class AlignBlockSimple(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.block[0].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.block[3].weight)

    def forward(self, x):
        return self.block(x)



class AlignBlockDepthSep(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            # depthwise
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            # pointwise
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),

            # depthwise
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            # pointwise
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.block[0].weight, mean=0.0, std=1e-3)  # first depthwise
        nn.init.normal_(self.block[1].weight, mean=0.0, std=1e-3)  # first pointwise
        nn.init.normal_(self.block[4].weight, mean=0.0, std=1e-3)  # second depthwise
        nn.init.zeros_(self.block[5].weight)                        # second pointwise
        
    def forward(self, x):
        return self.block(x)


class AlignBlock3DSimple(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # hidden = channels // 4

        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3,padding=1, bias=False),
            LayerNormNd(channels),
            nn.GELU(),

            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            LayerNormNd(channels),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.block[0].weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.block[3].weight, mean=0.0, std=1e-4)
 

    def forward(self, x):
        return self.block(x)


class AlignBlock3DDepthSep(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            # first depthwise separable conv
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),

            # second depthwise separable conv
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNormNd(channels),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.block[0].weight, mean=0.0, std=1e-4)  # first depthwise
        nn.init.normal_(self.block[1].weight, mean=0.0, std=1e-4)  # first pointwise
        nn.init.normal_(self.block[4].weight, mean=0.0, std=1e-4)  # second depthwise
        nn.init.normal_(self.block[5].weight, mean=0.0, std=1e-4)  # second pointwise

    def forward(self, x):
        return self.block(x)



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
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
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
    



class MultiLayerAlignAndFusionBaseline0(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=True,
        target_shape=512,
        feature_extraction=False,
        pretrain_decoder=False,
        freeze_adapter=False, 
        linear_prob=False,
        linear_prob_multi=False,
        decoder_prob=False,
        decoder_prob_multi=False,
        # interaction_indices=[2,5,8,11]
    ):
        super().__init__()

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode, Decoder, PatchDecodeBilinear

        self.dino_encoder = dino_encoder
        self.linear_prob = linear_prob
        self.pretrain_decoder = pretrain_decoder
        self.decoder_prob = decoder_prob
        self.decoder_prob_multi = decoder_prob_multi
        self.linear_prob_multi = linear_prob_multi


        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.target_shape = target_shape
        self.embed_dim = embed_dim
        self.interaction_indices=[2,5,8,11]

        # output_feature_channels = 768
        self.align2 = AlignBlockSimple(
            embed_dim
        )
        self.align5 = AlignBlockSimple(
            embed_dim, 
        )
        self.align8 = AlignBlockSimple(
            embed_dim, 
        )
        self.align11 = AlignBlockSimple(
            embed_dim,
        )

        if freeze_adapter:
            self.align2.requires_grad_(False)
            self.align2.eval()         

            self.align5.requires_grad_(False)
            self.align5.eval()         

            self.align8.requires_grad_(False)
            self.align8.eval()       

            self.align11.requires_grad_(False)
            self.align11.eval()          


        if not feature_extraction:
            if self.linear_prob:
                self.up_projection_reinit = PatchDecodeBilinear(
                    patch_embed_size
                )
                self.seg_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
            elif self.decoder_prob:
                self.up_projection_reinit = PatchDecode(
                    patch_embed_size, embed_dim, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection_reinit.ch[-2], num_classes, kernel_size=1)
            elif self.decoder_prob_multi:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection.ch[-2], num_classes, kernel_size=1)

            elif self.linear_prob_multi:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection.ch[-2], num_classes, kernel_size=1)
                self.up_projection.requires_grad_(False)
                self.up_projection.eval()

            else:
                raise NotImplementedError('do not support other finetuning tasks')
                
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
                )
        else:
            # pretraining part
            if self.pretrain_decoder:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )



    def forward(self, x, return_offset=False, return_final_only=False):
        assert x.shape[1] == 1

        x = x.repeat(1, 3, 1, 1)

        input_h = x.shape[-2]
        input_w = x.shape[-1]

        if input_w != self.target_shape or input_h != self.target_shape:
            x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
        x2, x5, x8, x11 = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)

        x2_offset = self.align2(x2)
        x5_offset = self.align5(x5)
        x8_offset = self.align8(x8)
        x11_offset = self.align11(x11)

        if return_final_only:
            return x11+x11_offset

        f_cat = torch.cat([x2+x2_offset, x5+x5_offset, x8+x8_offset, x11+x11_offset], dim=1)


        if self.feature_extraction:
            # pretrain
            if self.pretrain_decoder:
                dec_out = self.up_projection(f_cat)
            else:
                dec_out = f_cat

        else:
            # finetuneing
            if self.linear_prob or self.decoder_prob:
                dec_out = self.up_projection_reinit(x11+x11_offset)
            elif self.decoder_prob_multi or self.linear_prob_multi:
                dec_out = self.up_projection(f_cat)
            else: 
                raise NotImplementedError('Not supported yet')
            dec_out = self.seg_layer(dec_out)

        if return_offset:
            return dec_out, [x2_offset, x5_offset, x8_offset, x11_offset]
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")



class MultiLayerAlignAndFusionBaseline1(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=True,
        target_shape=512,
        feature_extraction=False,
        pretrain_decoder=False,
        freeze_adapter=False, 
        linear_prob=False,
        linear_prob_multi=False,
        decoder_prob=False,
        decoder_prob_multi=False,
        # interaction_indices=[2,5,8,11]
    ):
        super().__init__()

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode, Decoder, PatchDecodeBilinear

        self.dino_encoder = dino_encoder
        self.linear_prob = linear_prob
        self.pretrain_decoder = pretrain_decoder
        self.decoder_prob = decoder_prob
        self.decoder_prob_multi = decoder_prob_multi
        self.linear_prob_multi = linear_prob_multi
        self.embed_dim = embed_dim


        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()
        self.target_shape = target_shape
        self.embed_dim = embed_dim
        self.interaction_indices=[2,5,8,11]

        # output_feature_channels = 768
        self.align2 = AlignBlockDepthSep(
            embed_dim
        )
        self.align5 = AlignBlockDepthSep(
            embed_dim, 
        )
        self.align8 = AlignBlockDepthSep(
            embed_dim, 
        )
        self.align11 = AlignBlockDepthSep(
            embed_dim,
        )

        self.fusion_align = AlignBlockDepthSep(
            embed_dim*4
        )

        if freeze_adapter:
            self.align2.requires_grad_(False)
            self.align2.eval()         

            self.align5.requires_grad_(False)
            self.align5.eval()         

            self.align8.requires_grad_(False)
            self.align8.eval()       

            self.align11.requires_grad_(False)
            self.align11.eval()   

            self.fusion_align.requires_grad_(False)
            self.fusion_align.eval()     


        if not feature_extraction:
            if self.linear_prob:
                self.up_projection = PatchDecodeBilinear(
                    patch_embed_size
                )
                self.seg_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
            elif self.decoder_prob:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection.ch[-2], num_classes, kernel_size=1)
            elif self.decoder_prob_multi:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection.ch[-2], num_classes, kernel_size=1)

            elif self.linear_prob_multi:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv2d(self.up_projection.ch[-2], num_classes, kernel_size=1)
                self.up_projection.requires_grad_(False)
                self.up_projection.eval()

            else:
                raise NotImplementedError('do not support other finetuning tasks')
                
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
                )
        else:
            # pretraining part
            if self.pretrain_decoder:
                self.up_projection = PatchDecode(
                    patch_embed_size, embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )



    def forward(self, x, return_offset=False, return_final_only=False):
        assert x.shape[1] == 1

        x = x.repeat(1, 3, 1, 1)

        input_h = x.shape[-2]
        input_w = x.shape[-1]

        if input_w != self.target_shape or input_h != self.target_shape:
            x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
        x2, x5, x8, x11 = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)

        x2_offset = self.align2(x2)
        x5_offset = self.align5(x5)
        x8_offset = self.align8(x8)
        x11_offset = self.align11(x11)

        f_cat = torch.cat([x2+x2_offset, x5+x5_offset, x8+x8_offset, x11+x11_offset], dim=1)
        fusion_offset = self.fusion_align(f_cat)
        x2_fusion_offset, x5_fusion_offset, x8_fusion_offset, x11_fusion_offset = \
            torch.chunk(fusion_offset, 4, dim=1)
        f_cat = f_cat + fusion_offset

        if return_final_only:
            return x11+x11_offset+x11_fusion_offset



        if self.feature_extraction:
            # pretrain
            if self.pretrain_decoder:
                dec_out = self.up_projection(f_cat)
            else:
                dec_out = f_cat

        else:
            # finetuneing
            if self.linear_prob or self.decoder_prob:
                dec_out = self.up_projection(x11+x11_offset+x11_fusion_offset)
            elif self.decoder_prob_multi or self.linear_prob_multi:
                dec_out = self.up_projection(f_cat)
            else: 
                raise NotImplementedError('Not supported yet')
            dec_out = self.seg_layer(dec_out)

        if return_offset:
            return dec_out, [x2_offset, x5_offset, x8_offset, x11_offset, x2_fusion_offset, x5_fusion_offset, x8_fusion_offset, x11_fusion_offset]
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")





# class MultiLayerAlignAndFusionBaseline1(AbstractDynamicNetworkArchitectures):

#     def __init__(
#         self,
#         embed_dim: int,
#         patch_embed_size: int,
#         num_classes: int,
#         decoder_norm=LayerNormNd,
#         decoder_act=nn.GELU,
#         dino_encoder=None,
#         freeze_backbone=True,
#         freeze_adapter=False
#         target_shape=512,
#         feature_extraction=False, # otherwise finetune also the whole DINOv3
#         linear_prob=False,
#         decoder_prob=False,
#         decoder_prob_multi=False
#     ):
#         super().__init__()

#         from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode, Decoder, PatchDecodeBilinear

#         self.dino_encoder = dino_encoder
#         self.linear_prob = linear_prob
#         self.decoder_linear_prob = decoder_linear_prob
#         self.feature_extraction =feature_extraction
#         if freeze_backbone:
#             self.dino_encoder.requires_grad_(False)
#             self.dino_encoder.eval()

#         self.decoder = Decoder()
#         self.target_shape = target_shape
#         self.embed_dim = embed_dim
#         self.interaction_indices=[2,5,8,11]

#         output_feature_channels = 768
#         self.align2 = AlignBlockSimple(
#             embed_dim
#         )
#         self.align5 = AlignBlockSimple(
#             embed_dim, 
#         )
#         self.align8 = AlignBlockSimple(
#             embed_dim, 
#         )
#         self.align11 = AlignBlockSimple(
#             embed_dim,
#         )
#         if freeze_adapter:
#             self.align2.requires_grad_(False)
#             self.align2.eval()      

#             self.align5.requires_grad_(False)
#             self.align5.eval()      

#             self.align8.requires_grad_(False)
#             self.align8.eval()      

#             self.align11.requires_grad_(False)
#             self.align11.eval()      



#         if not feature_extraction:      
#             if not linear_prob and not decoder_linear_prob:          
#                 self.up_projection = PatchDecode(
#                         patch_embed_size, output_feature_channels*4, 128, norm=decoder_norm, activation=decoder_act
#                     )
#                 self.seg_layer = nn.Conv2d(128, num_classes, kernel_size=1)
#             elif linear_prob:
#                 assert not decoder_linear_prob,'linear prob the final layer of model'
#                 self.up_projection = PatchDecodeBilinear(
#                         patch_embed_size
#                     )
#                 self.seg_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
#             elif decoder_linear_prob:
#                 self.up_projection = PatchDecode(
#                         patch_embed_size, output_feature_channels*4, 128, norm=decoder_norm, activation=decoder_act
#                     )
#                 self.up_projection.requires_grad_(False)
#                 self.up_projection.eval()      
#                 self.seg_layer = nn.Conv2d(128, num_classes, kernel_size=1)           
                


#         else:
#             self.up_projection = PatchDecode(
#                 patch_embed_size, output_feature_channels*4, 128, norm=decoder_norm, activation=decoder_act
#             )



#     def forward(self, x, return_norm=False):
#         assert x.shape[1] == 1

#         x = x.repeat(1, 3, 1, 1)

#         input_h = x.shape[-2]
#         input_w = x.shape[-1]

#         if input_w != self.target_shape or input_h != self.target_shape:
#             x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
#         x2, x5, x8, x11 = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)

#         x2_offset = self.align2(x2)
#         x5_offset = self.align5(x5)
#         x8_offset = self.align8(x8)
#         x11_offset = self.align11(x11)
#         x2 = x2 + x2_offset
#         x5 = x5 + x5_offset
#         x8 = x8 + x8_offset
#         x11 = x11 + x11_offset

#         x2_offset_norm = x2_offset.abs().mean()
#         x5_offset_norm = x5_offset.abs().mean()
#         x8_offset_norm = x8_offset.abs().mean()
#         x11_offset_norm = x11_offset.abs().mean()

#         f_cat = torch.cat([x2, x5, x8, x11], dim=1)

#         if not self.feature_extraction:
#             if self.linear_prob :
#                 dec_out = self.up_projection(x11)
#                 dec_out = self.seg_layer(dec_out)

#             elif self.decoder_linear_prob:
#                 dec_out = self.up_projection(f_cat)
#                 dec_out = self.seg_layer(dec_out)
#             else:
#                 dec_out = self.up_projection(f_cat)
#                 # dec_out = self.up_projection(f_cat)
#                 # if self.linear_prob:
#                 dec_out = self.seg_layer(dec_out)
#                     # print("dec_out", dec_out.shape)
#         else:
#             dec_out = self.up_projection(f_cat)

#         if return_norm:
#             return dec_out, (x2_offset_norm + x5_offset_norm + x8_offset_norm + x11_offset_norm)
#         else:
#             return dec_out

#     def compute_conv_feature_map_size(self, input_size):
#         raise NotImplementedError("yuck")


# class MultiLayerAlignAndFusionBaseline2(AbstractDynamicNetworkArchitectures):

#     def __init__(
#         self,
#         embed_dim: int,
#         patch_embed_size: int,
#         num_classes: int,
#         decoder_norm=LayerNormNd,
#         decoder_act=nn.GELU,
#         dino_encoder=None,
#         freeze_backbone=True,
#         target_shape=512,
#         feature_extraction=False, # for pretraining, true for segmentation with decoder
#         linear_prob=False,
#         # interaction_indices=[2,5,8,11]
#     ):
#         super().__init__()

#         from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode, Decoder, PatchDecodeBilinear

#         self.dino_encoder = dino_encoder
#         self.linear_prob = linear_prob
#         self.feature_extraction =feature_extraction
#         if freeze_backbone:
#             self.dino_encoder.requires_grad_(False)
#             self.dino_encoder.eval()

#         self.decoder = Decoder()
#         self.target_shape = target_shape
#         self.embed_dim = embed_dim
#         self.interaction_indices=[2,5,8,11]

#         output_feature_channels = 768
#         self.align2 = AlignBlockSimple(
#             embed_dim
#         )
#         self.align5 = AlignBlockSimple(
#             embed_dim, 
#         )
#         self.align8 = AlignBlockSimple(
#             embed_dim, 
#         )
#         self.align11 = AlignBlockSimple(
#             embed_dim,
#         )



#         if not feature_extraction:  
                          
#             self.up_projection = PatchDecode(
#                     patch_embed_size, output_feature_channels*4, 0, norm=decoder_norm, activation=decoder_act
#                 )
#             self.seg_layer = nn.Conv2d(192, num_classes, kernel_size=1)

#         else:
#             self.up_projection = PatchDecode(
#                 patch_embed_size, output_feature_channels*4, 0, norm=decoder_norm, activation=decoder_act
#             )



#     def forward(self, x, ret_mask=False):
#         assert x.shape[1] == 1

#         x = x.repeat(1, 3, 1, 1)

#         input_h = x.shape[-2]
#         input_w = x.shape[-1]

#         if input_w != self.target_shape or input_h != self.target_shape:
#             x = nn.functional.interpolate(x, size=(self.target_shape, self.target_shape), mode='bilinear', align_corners=False)
#         x2, x5, x8, x11 = self.dino_encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=True)

#         x2_offset = self.align2(x2)
#         x5_offset = self.align5(x5)
#         x8_offset = self.align8(x8)
#         x11_offset = self.align11(x11)
#         x2 = x2 + x2_offset
#         x5 = x5 + x5_offset
#         x8 = x8 + x8_offset
#         x11 = x11 + x11_offset

#         x2_offset_norm = x2_offset.abs().mean()
#         x5_offset_norm = x5_offset.abs().mean()
#         x8_offset_norm = x8_offset.abs().mean()
#         x11_offset_norm = x11_offset.abs().mean()

#         f_cat = torch.cat([x2, x5, x8, x11], dim=1)

#         if not self.feature_extraction:
#             dec_out = self.up_projection(f_cat)
#             # dec_out = self.up_projection(f_cat)
#             # if self.linear_prob:
#             dec_out = self.seg_layer(dec_out)
#                 # print("dec_out", dec_out.shape)
#         else:
#             dec_out = self.up_projection(f_cat)

#         return dec_out, (x2_offset_norm + x5_offset_norm + x8_offset_norm + x11_offset_norm)

#     def compute_conv_feature_map_size(self, input_size):
#         raise NotImplementedError("yuck")




class MultiLayerAlignAndFusion3DBaseline0(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=True,
        feature_extraction=False,
        linear_prob=False,
        decoder_prob=False,
        decoder_prob_multi=False,
        target_shape_for_dino = None,
        freeze_adapter=False
        # interaction_indices=[2,5,8,11]
    ):
        super().__init__()

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode3D, Decoder, PatchDecodeBilinear, PatchDecodeTrilinear

        self.dino_encoder = dino_encoder
        self.target_shape_for_dino =target_shape_for_dino
        self.linear_prob = linear_prob
        self.decoder_prob = decoder_prob
        self.decoder_prob_multi = decoder_prob_multi
        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()

        self.embed_dim = embed_dim
        self.interaction_indices=[2,5,8,11]

        output_feature_channels = embed_dim
        self.align2 = AlignBlock3DSimple(
            embed_dim
        )
        self.align5 = AlignBlock3DSimple(
            embed_dim, 
        )
        self.align8 = AlignBlock3DSimple(
            embed_dim, 
        )
        self.align11 = AlignBlock3DSimple(
            embed_dim,
        )

        if freeze_adapter:
            self.align2.requires_grad_(False)
            self.align2.eval()     

            self.align5.requires_grad_(False)
            self.align5.eval()     

            self.align8.requires_grad_(False)
            self.align8.eval()     

            self.align11.requires_grad_(False)
            self.align11.eval()     


        if not feature_extraction:
            if self.linear_prob:
                self.up_projection = PatchDecodeTrilinear(
                    patch_embed_size
                )
                self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
            elif self.decoder_prob:
                self.up_projection = PatchDecode3D(
                     (1, patch_embed_size, patch_embed_size), embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
                )
            elif self.decoder_prob_multi:
                self.up_projection = PatchDecode3D(
                   (1, patch_embed_size, patch_embed_size), embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
                )         
                # output_feature_channels = output_feature_channels*4

                # self.seg_layer = nn.Conv2d(output_feature_channels, num_classes, kernel_size=1)
        else:
            
            self.up_projection = PatchDecode3D(
                (1, patch_embed_size, patch_embed_size), embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
            )

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
            # print("Finsih dino process")
            for dino_features in dino_features_list:
                # print("dino_features 0 ", dino_features.shape)
                feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
                dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
                dino_features = dino_features.permute(0,2,1,3,4,)
                # print("dino_features 1", dino_features.shape)
                dino_features_list_new.append(dino_features)
        return dino_features_list_new


    def forward(self, x, return_offset=False):
        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3, 1, 1, 1)
        # print("x shape", x.shape)
        x2, x5, x8, x11 = self.prepare_dino_features(x)
        # print("x2", x2.shape, 'x5', x5.shape, 'x8', x8.shape, 'x11', x11.shape)
        
        x2_offset = self.align2(x2)
        x5_offset = self.align5(x5)
        x8_offset = self.align8(x8)
        x11_offset = self.align11(x11)

        # if return_final_only:
        #     return x11 + x11_offset

        f_cat = torch.cat([x2+x2_offset, x5+x5_offset, x8+x8_offset, x11+x11_offset], dim=1)
      
        x2_offset_norm = x2_offset.abs().mean()
        x5_offset_norm = x5_offset.abs().mean()
        x8_offset_norm = x8_offset.abs().mean()
        x11_offset_norm = x11_offset.abs().mean()
        # print("debug norm", x2_offset_norm+x5_offset_norm+x8_offset_norm+x11_offset_norm )

        # dec_out = self.up_projection(f_cat)

        if not self.feature_extraction:
        #     dec_out = self.up_projection(f_cat)
        #     # if self.linear_prob:
        #     #     dec_out = self.seg_layer(dec_out)
        #         # print("dec_out", dec_out.shape)
        # else:
            if self.linear_prob:
                dec_out = self.up_projection(x11+x11_offset)
                dec_out = self.seg_layer(dec_out)
            elif self.decoder_prob:
                dec_out = self.up_projection(x11+x11_offset)

            elif self.decoder_prob_multi:
                dec_out = self.up_projection(f_cat)
            else:
                dec_out = f_cat # no upsample
        else:
            dec_out = f_cat
        if not return_offset:
            return dec_out
        else:
            return dec_out, [x2_offset, x5_offset, x8_offset, x11_offset]

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")




class MultiLayerAlignAndFusion3DBaseline01(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=True,
        feature_extraction=False,
        linear_prob=False,
        decoder_prob=False,
        linear_prob_multi=False,
        decoder_prob_multi=False,
        target_shape_for_dino = None,
        freeze_adapter=False,
        pretrain_decoder=False
    ):
        super().__init__()

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode3D, Decoder, PatchDecodeBilinear, PatchDecodeTrilinear

        self.dino_encoder = dino_encoder
        self.target_shape_for_dino =target_shape_for_dino
        self.linear_prob = linear_prob
        self.decoder_prob = decoder_prob
        self.decoder_prob_multi = decoder_prob_multi
        self.linear_prob_multi = linear_prob_multi
        self.pretrain_decoder = pretrain_decoder


        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()

        self.embed_dim = embed_dim
        self.interaction_indices=[2,5,8,11]

        output_feature_channels = embed_dim
        self.align2 = AlignBlock3DDepthSep(
            embed_dim
        )
        self.align5 = AlignBlock3DDepthSep(
            embed_dim, 
        )
        self.align8 = AlignBlock3DDepthSep(
            embed_dim, 
        )
        self.align11 = AlignBlock3DDepthSep(
            embed_dim,
        )
        if not self.pretrain_decoder:
            self.fusion_align = AlignBlock3DDepthSep(
                embed_dim*4
            )

        if freeze_adapter:
            self.align2.requires_grad_(False)
            self.align2.eval()     

            self.align5.requires_grad_(False)
            self.align5.eval()     

            self.align8.requires_grad_(False)
            self.align8.eval()     

            self.align11.requires_grad_(False)
            self.align11.eval()     

            self.fusion_align.requires_grad_(False)
            self.fusion_align.eval()     


        if not feature_extraction:
            if self.linear_prob:
                self.up_projection_reinit = PatchDecodeTrilinear(
                    patch_embed_size
                )
                self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
            elif self.decoder_prob:

                self.up_projection_reinit = PatchDecode3D(
                     (1, patch_embed_size, patch_embed_size), embed_dim, 0, norm=decoder_norm, activation=decoder_act
                )
                self.seg_layer = nn.Conv3d(self.up_projection_reinit.channels[-2], num_classes, kernel_size=1)


            elif self.decoder_prob_multi:
                self.up_projection = PatchDecode3D(
                   (1, patch_embed_size, patch_embed_size), embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )         

                self.seg_layer = nn.Conv2d(self.up_projection_reinit.channels[-2], num_classes, kernel_size=1)

            elif self.linear_prob_multi:
                self.up_projection = PatchDecode3D(
                (1, patch_embed_size, patch_embed_size), embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )

                self.seg_layer = nn.Conv3d(self.up_projection.channels[-2],num_classes, kernel_size=1)
                self.up_projection.requires_grad_(False)
                self.up_projection.eval()

        else:
            if self.pretrain_decoder:
                self.up_projection = PatchDecode3D(
                    (1, patch_embed_size, patch_embed_size), embed_dim*4, 0, norm=decoder_norm, activation=decoder_act
                )

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
            # print("Finsih dino process")
            for dino_features in dino_features_list:
                # print("dino_features 0 ", dino_features.shape)
                feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
                dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
                dino_features = dino_features.permute(0,2,1,3,4,)
                # print("dino_features 1", dino_features.shape)
                dino_features_list_new.append(dino_features)
        return dino_features_list_new


    def forward(self, x, return_offset=False, return_final_only=False):
        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3, 1, 1, 1)
        # print("x shape", x.shape)
        x2, x5, x8, x11 = self.prepare_dino_features(x)
        # print("x2", x2.shape, 'x5', x5.shape, 'x8', x8.shape, 'x11', x11.shape)
        
        x2_offset = self.align2(x2)
        x5_offset = self.align5(x5)
        x8_offset = self.align8(x8)
        x11_offset = self.align11(x11)
        f_cat = torch.cat([x2+x2_offset, x5+x5_offset, x8+x8_offset, x11+x11_offset], dim=1)

        if not self.pretrain_decoder:
            fusion_offset = self.fusion_align(f_cat)
            x2_fusion_offset, x5_fusion_offset, x8_fusion_offset, x11_fusion_offset = \
                torch.chunk(fusion_offset, 4, dim=1)
        else:
            x2_fusion_offset, x5_fusion_offset, x8_fusion_offset, x11_fusion_offset = (
                torch.zeros_like(x2),
                torch.zeros_like(x5),
                torch.zeros_like(x8),
                torch.zeros_like(x11),
            )
        if return_final_only:
            return x11 + x11_offset + x2_fusion_offset



        if self.feature_extraction:
            # pretrain
            if self.pretrain_decoder:
                dec_out = self.up_projection(f_cat)
            else:
                dec_out = f_cat + fusion_offset

        else:
            # finetuneing
            if self.linear_prob or self.decoder_prob:
                dec_out = self.up_projection(x11+x11_offset+x11_fusion_offset)
            elif self.decoder_prob_multi or self.linear_prob_multi:
                dec_out = self.up_projection(f_cat)
            else: 
                raise NotImplementedError('Not supported yet')
            dec_out = self.seg_layer(dec_out)

        print("dec_out", dec_out.shape)
        if return_offset:
            return dec_out, [x2_offset, x5_offset, x8_offset, x11_offset, x2_fusion_offset, x5_fusion_offset, x8_fusion_offset, x11_fusion_offset]
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")






class MultiLayerAlignAndFusion3DBaseline1(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder=None,
        freeze_backbone=True,
        feature_extraction=False,
        linear_prob=False,
        decoder_prob=False,
        decoder_prob_multi=False,
        target_shape_for_dino = None,
        freeze_adapter=False
        # interaction_indices=[2,5,8,11]
    ):
        super().__init__()

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PatchDecode3D, Decoder, PatchDecodeBilinear, PatchDecodeTrilinear

        self.dino_encoder = dino_encoder
        self.target_shape_for_dino =target_shape_for_dino
        self.linear_prob = linear_prob
        self.decoder_prob = decoder_prob
        self.decoder_prob_multi = decoder_prob_multi
        self.feature_extraction =feature_extraction
        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()

        self.decoder = Decoder()

        self.embed_dim = embed_dim
        self.interaction_indices=[2,5,8,11]

        output_feature_channels = embed_dim
        self.align2 = AlignBlock3DSimple(
            embed_dim
        )
        self.align5 = AlignBlock3DSimple(
            embed_dim, 
        )
        self.align8 = AlignBlock3DSimple(
            embed_dim, 
        )
        self.align11 = AlignBlock3DSimple(
            embed_dim,
        )

        if freeze_adapter:
            self.align2.requires_grad_(False)
            self.align2.eval()     

            self.align5.requires_grad_(False)
            self.align5.eval()     

            self.align8.requires_grad_(False)
            self.align8.eval()     

            self.align11.requires_grad_(False)
            self.align11.eval()     


        if not feature_extraction:
            if self.linear_prob:
                self.up_projection = PatchDecodeTrilinear(
                    patch_embed_size
                )
                self.seg_layer = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
            elif self.decoder_prob:
                self.up_projection = PatchDecode3D(
                     (1, patch_embed_size, patch_embed_size), embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
                )
            elif self.decoder_prob_multi:
                self.up_projection = PatchDecode3D(
                   (1, patch_embed_size, patch_embed_size), embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
                )         
                # output_feature_channels = output_feature_channels*4

                # self.seg_layer = nn.Conv2d(output_feature_channels, num_classes, kernel_size=1)
        else:
            
            self.up_projection = PatchDecode3D(
                (1, patch_embed_size, patch_embed_size), embed_dim*4, num_classes, norm=decoder_norm, activation=decoder_act
            )

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
            # print("Finsih dino process")
            for dino_features in dino_features_list:
                # print("dino_features 0 ", dino_features.shape)
                feature_c, feature_h, feature_w = dino_features.shape[1], dino_features.shape[2], dino_features.shape[3]
                dino_features = dino_features.reshape(b,d, feature_c,feature_h,feature_w)
                dino_features = dino_features.permute(0,2,1,3,4,)
                # print("dino_features 1", dino_features.shape)
                dino_features_list_new.append(dino_features)
        return dino_features_list_new


    def forward(self, x, return_offset=False, return_final_only=False):
        assert x.shape[1] == 1
        assert len(x.shape) == 5

        x = x.repeat(1, 3, 1, 1, 1)
        # print("x shape", x.shape)
        x2, x5, x8, x11 = self.prepare_dino_features(x)
        # print("x2", x2.shape, 'x5', x5.shape, 'x8', x8.shape, 'x11', x11.shape)
        
        x2_offset = self.align2(x2)
        x5_offset = self.align5(x5)
        x8_offset = self.align8(x8)
        x11_offset = self.align11(x11)

        if return_final_only:
            return x11 + x11_offset

        f_cat = torch.cat([x2+x2_offset, x5+x5_offset, x8+x8_offset, x11+x11_offset], dim=1)
      
        x2_offset_norm = x2_offset.abs().mean()
        x5_offset_norm = x5_offset.abs().mean()
        x8_offset_norm = x8_offset.abs().mean()
        x11_offset_norm = x11_offset.abs().mean()
        # print("debug norm", x2_offset_norm+x5_offset_norm+x8_offset_norm+x11_offset_norm )

        # dec_out = self.up_projection(f_cat)

        if not self.feature_extraction:
        #     dec_out = self.up_projection(f_cat)
        #     # if self.linear_prob:
        #     #     dec_out = self.seg_layer(dec_out)
        #         # print("dec_out", dec_out.shape)
        # else:
            if self.linear_prob:
                dec_out = self.up_projection(x11+x11_offset)
                dec_out = self.seg_layer(dec_out)
            elif self.decoder_prob:
                dec_out = self.up_projection(x11+x11_offset)

            elif self.decoder_prob_multi:
                dec_out = self.up_projection(f_cat)
            else:
                dec_out = f_cat # no upsample
        else:
            dec_out = self.up_projection(f_cat)
            
        if not return_offset:
            return dec_out
        else:
            return dec_out, [x2_offset, x5_offset, x8_offset, x11_offset]

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")

