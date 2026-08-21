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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

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



class Multiscale_2d_encoder_only(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        freeze_adapter=False,
        interaction_indices=[2,5,8,11],
        upsample_factor=1,
        proj_head=None,
    ):
        """
        dino encoder + no decoder
        """
        super().__init__()
        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            if not freeze_adapter:
                for name, p in self.dino_encoder.named_parameters():
                    if "Inner_Adapter" in name:
                        p.requires_grad = True   
                        print("Name require grad", name)   


            self.dino_encoder.eval()

            if not freeze_adapter:
                for name, m in self.dino_encoder.named_modules():
                    if "Inner_Adapter" in name:
                        print("Name for .train()", name)
                        m.train()
                    
        self.decoder = Decoder()
        # self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices
        self.upsample_factor = upsample_factor
        self.num_classes =num_classes
        if num_classes > 0:
            self.seg_head = nn.Conv2d(embed_dim*4, num_classes, kernel_size=1)

        if proj_head is not None:
            self.proj_head = nn.Sequential(
                        nn.Conv2d(embed_dim*4, 1024, 1, bias=False),
                        LayerNormNd(1024),
                        nn.GELU(),
                        nn.Conv2d(1024, 512, 1, bias=False),
                    )
        else:
            self.proj_head =None

            


    def forward(self, x, ret_mask=False, return_dino_feature=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        if self.upsample_factor > 1:
            # bilinear upsample here
            hier = F.interpolate(
                        hier,
                        scale_factor=self.upsample_factor,
                        mode="bilinear",
                        align_corners=False,
                    )
        if self.num_classes > 0:
            # finetuning model, map to output class
            output = self.seg_head(hier)


        if self.proj_head is not None:
            output = self.proj_head(hier)

        if return_dino_feature:
            return output, hier
        else:
            return output

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")




class Multiscale_2d_conv_decoder(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        freeze_backbone=False,
        freeze_adapter=False,
        interaction_indices=[2,5,8,11],
        proj_head=None,
        linear_prob=False,
        decoder_prob=False,
    ):
        """
        dino encoder + 2d decoder 
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder

        if freeze_backbone:
            self.dino_encoder.requires_grad_(False)
            if not freeze_adapter:
                for name, p in self.dino_encoder.named_parameters():
                    if "Inner_Adapter" in name:
                        p.requires_grad = True   
                        print("Name require grad", name)   


            self.dino_encoder.eval()

            if not freeze_adapter:
                for name, m in self.dino_encoder.named_modules():
                    if "Inner_Adapter" in name:
                        print("Name for .train()", name)
                        m.train()

        self.linear_prob = linear_prob
        if linear_prob:
            # also freeze the decoder
            assert num_classes>0
            self.up_projection.requires_grad_(False)
            self.up_projection.eval()
            self.seg_head = nn.Conv2d(self.up_projection.ch[-2], num_classes, 1)
        else:
            self.seg_head = None 


        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices
        if proj_head is not None:
            self.proj_head = nn.Sequential(
                        nn.Conv2d(self.up_projection.ch[-2], self.up_projection.ch[-2], 3, padding=1),
                        LayerNormNd(self.up_projection.ch[-2]),
                        nn.GELU(),
                        nn.Conv2d(self.up_projection.ch[-2], 512, 1),
                    )
            
            #  nn.Conv2d(self.up_projection.ch[-2], proj_head, kernel_size=1)
        else:
            self.proj_head = None


    def forward(self, x, ret_mask=False, return_dino_feature=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.up_projection(hier)
        if self.proj_head is not None:
            dec_out = self.proj_head(dec_out)

        if self.seg_head is not None:
            dec_out = self.seg_head(dec_out)


        if return_dino_feature:
            return dec_out, hier
        else:
            return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


############################### finetuning ###########################
