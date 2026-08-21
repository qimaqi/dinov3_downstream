import os
import sys
from typing import List, Tuple, Union

import torch
from torch import nn

# Keep local dinov3 imports working when nnU-Net imports this trainer module directly.
dino_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3")
if dino_root not in sys.path:
    sys.path.append(dino_root)

from nnunetv2.training.nnUNetTrainer.dinomed2dFintuner import dinomedFintuner, dinov3_base_encoder_primus_decoder_Trainer_thin_3d_aug_noflip


DINOV3_CHECKPOINT = (
    "/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def _load_dinov3_state(model: nn.Module, checkpoint_path: str = DINOV3_CHECKPOINT) -> None:
    chkpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = chkpt.get("teacher", chkpt)
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if "ibot" not in k and "dino_head" not in k
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Load DINOv3 SliceWiseDinoVisionTransformer", "missing", missing, "unexpected", unexpected)


def _set_encoder_mode_for_cross_slice_tuning(encoder: nn.Module) -> None:
    encoder.eval()
    for module_name, module in encoder.named_modules():
        if "cross_slice_mixer" in module_name:
            module.train()


def _freeze_encoder_for_tuning(network: nn.Module, train_layer_norm: bool = False) -> None:
    encoder = getattr(network, "dino_encoder", None)
    if encoder is None:
        raise AttributeError("Expected network to expose a dino_encoder attribute.")

    encoder.requires_grad_(False)
    _set_encoder_mode_for_cross_slice_tuning(encoder)

    trainable_names = []
    for name, param in encoder.named_parameters():
        is_cross_slice_param = "cross_slice_mixer" in name
        is_norm_param = train_layer_norm and "norm" in name.lower()
        if is_cross_slice_param or is_norm_param:
            param.requires_grad_(True)
            trainable_names.append(name)

    if not any("cross_slice_mixer" in name for name in trainable_names):
        raise RuntimeError("No cross_slice_mixer parameters were found to unfreeze in the DINO encoder.")
    if train_layer_norm and not any("norm" in name.lower() for name in trainable_names):
        raise RuntimeError("No LayerNorm/norm parameters were found to unfreeze in the DINO encoder.")

    print("Trainable encoder parameters:")
    for name in trainable_names:
        print(f"  {name}")


def _freeze_encoder_without_cross_slice_mixer(network: nn.Module, train_layer_norm: bool = False) -> None:
    encoder = getattr(network, "dino_encoder", None)
    if encoder is None:
        raise AttributeError("Expected network to expose a dino_encoder attribute.")

    encoder.eval()
    encoder.requires_grad_(False)

    trainable_names = []
    if train_layer_norm:
        for name, param in encoder.named_parameters():
            if "norm" in name.lower():
                param.requires_grad_(True)
                trainable_names.append(name)

        if not trainable_names:
            raise RuntimeError("No LayerNorm/norm parameters were found to unfreeze in the DINO encoder.")

    if trainable_names:
        print("Trainable encoder parameters:")
        for name in trainable_names:
            print(f"  {name}")
    else:
        print("DINO encoder is fully frozen; no cross_slice_mixer parameters are used.")


def _set_default_training_hparams(trainer: dinomedFintuner) -> None:
    trainer.initial_lr = 3e-4
    trainer.vit_lr = 1e-4
    trainer.weight_decay = 5e-2
    trainer.vit_weight_decay = 5e-2
    trainer.oversample_foreground_percent = 0.33
    trainer.num_iterations_per_epoch = 250
    trainer.num_val_iterations_per_epoch = 50
    trainer.warmup_epochs = 0
    trainer.num_epochs = 1000
    trainer.current_epoch = 0
    trainer.enable_deep_supervision = False


def _configure_adamw_split_optimizer(trainer: dinomedFintuner):
    vit_params = []
    other_params = []
    for name, param in trainer.network.named_parameters():
        if "dino_encoder" in name:
            vit_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": trainer.initial_lr, "weight_decay": trainer.weight_decay},
            {"params": vit_params, "lr": trainer.vit_lr, "weight_decay": trainer.vit_weight_decay},
        ],
        betas=(0.9, 0.98),
    )

    print("=== Trainable Parameters ===")
    total_trainable = 0
    for name, param in trainer.network.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_trainable += num_params
            print(f"{name:60s} | shape={tuple(param.shape)} | params={num_params:,}")
    print(f"\nTotal trainable parameters: {total_trainable:,}")

    total_iters = trainer.num_epochs

    def lr_lambda(current_iter):
        if current_iter < trainer.warmup_epochs:
            return (1e-6 + (trainer.initial_lr - 1e-6) * (current_iter / trainer.warmup_epochs)) / trainer.initial_lr
        progress = (current_iter - trainer.warmup_epochs) / (total_iters - trainer.warmup_epochs)
        return (1 - progress) ** 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, lr_scheduler


class Linear3DCrossSliceTuningMixin:
    def train(self, mode: bool = True):
        super().train(mode)
        _set_encoder_mode_for_cross_slice_tuning(self.dino_encoder)
        return self




class crosslayer_dinomed3dFintuner(dinov3_base_encoder_primus_decoder_Trainer_thin_3d_aug_noflip):
    """nnU-Net trainer for MedDINOv3 slice-wise 3D fine-tuning.

    The network is Linear3D(dino_encoder=SliceWiseDinoVisionTransformer). The DINO
    encoder is frozen except for its cross-slice token mixer parameters; the Linear3D
    decoder remains trainable.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        _set_default_training_hparams(self)

    def configure_optimizers(self):
        return _configure_adamw_split_optimizer(self)

    train_encoder_layer_norm = False
    decoder_name = "Linear3D"

    @classmethod
    def build_network_architecture(
        cls,
        patch_size: tuple,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.slicewise_decoder import Decoder3D, Linear3D
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import (
            SliceWiseDinoVisionTransformer,
        )
        print("==== patch_size ====", patch_size)
        max_slices = patch_size[0] #16 # int(max(patch_size)) if patch_size is not None and len(patch_size) > 0 else 256
        dino_encoder = SliceWiseDinoVisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias=True,
            mask_k_bias=True,
            cross_slice_mixer="mlp_mixer",
            slice_mixer_num_slices=max_slices,
            slice_mixer_bottleneck_dim=128,
            slice_mixer_num_heads=4,
            slice_mixer_mlp_ratio=2.0,
            slice_mixer_init_gamma=0.0,
            use_slice_pos_embed=True,
            use_token_type_embed=True,
        )
        _load_dinov3_state(dino_encoder)

        decoder_cls = {
            "Linear3D": Linear3D,
            "Decoder3D": Decoder3D,
        }[cls.decoder_name]

        class CrossSliceTuningDecoder(Linear3DCrossSliceTuningMixin, decoder_cls):
            pass

        network = CrossSliceTuningDecoder(
            embed_dim=768,
            patch_embed_size=16,
            num_classes=num_output_channels,
            dino_encoder=dino_encoder,
            freeze_backbone=False,
        )
        _freeze_encoder_for_tuning(network, train_layer_norm=cls.train_encoder_layer_norm)
        return network

# no cross layer, used for debug
class dinomed3dFintuner(dinov3_base_encoder_primus_decoder_Trainer_thin_3d_aug_noflip):
    """nnU-Net trainer for MedDINOv3 slice-wise 3D fine-tuning.

    The network is Linear3D(dino_encoder=SliceWiseDinoVisionTransformer). The DINO
    encoder has no cross-slice mixer and is frozen; the Linear3D decoder remains
    trainable.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        _set_default_training_hparams(self)

    def configure_optimizers(self):
        return _configure_adamw_split_optimizer(self)

    train_encoder_layer_norm = False
    decoder_name = "Linear3D"

    @classmethod
    def build_network_architecture(
        cls,
        patch_size: tuple,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.slicewise_decoder import Decoder3D, Linear3D
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import (
            SliceWiseDinoVisionTransformer,
        )
        print("==== patch_size ====", patch_size)
        max_slices = patch_size[0] #16 # int(max(patch_size)) if patch_size is not None and len(patch_size) > 0 else 256
        dino_encoder = SliceWiseDinoVisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias=True,
            mask_k_bias=True,
            cross_slice_mixer=None,
            slice_mixer_num_slices=max_slices,
            slice_mixer_bottleneck_dim=128,
            slice_mixer_num_heads=4,
            slice_mixer_mlp_ratio=2.0,
            slice_mixer_init_gamma=0.0,
            use_slice_pos_embed=True,
            use_token_type_embed=True,
        )
        _load_dinov3_state(dino_encoder)

        decoder_cls = {
            "Linear3D": Linear3D,
            "Decoder3D": Decoder3D,
        }[cls.decoder_name]

        class FrozenDinoDecoder(decoder_cls):
            pass

        network = FrozenDinoDecoder(
            embed_dim=768,
            patch_embed_size=16,
            num_classes=num_output_channels,
            dino_encoder=dino_encoder,
            freeze_backbone=False,
        )
        _freeze_encoder_without_cross_slice_mixer(network, train_layer_norm=cls.train_encoder_layer_norm)
        return network




class mlpcrosslayer_dinomed3dFintuner_linear_layernorm(crosslayer_dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with cross-slice mixer and encoder LayerNorm parameters trainable."""

    train_encoder_layer_norm = True

class mlpcrosslayer_dinomed3dFintuner_linear(crosslayer_dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Linear3D"


class mlpcrosslayer_dinomed3dFintuner_decoder(crosslayer_dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Decoder3D"


class dinomed3dFintuner_decoder(dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Decoder3D"


class dinomed3dFintuner_decoder_layernorm(dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Decoder3D"
    train_encoder_layer_norm = True



class dinomed3dFintuner_linear(dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Linear3D"


class dinomed3dFintuner_linear_layernorm(dinomed3dFintuner):
    """MedDINOv3 3D fine-tuning with MLP cross-slice mixer and Decoder3D decoder."""

    decoder_name = "Linear3D"
    train_encoder_layer_norm = True
