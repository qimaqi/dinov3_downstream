# Experiment Code Map

This file is a lightweight navigation index for the experimental DINO/MedDINO trainer and model files. It does not define behavior; it points from class names to source locations.

For shell-job-to-trainer mapping, see `sh_jobs/amos/README.md`.

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`

Main nn-U-Net trainer variants for DINOv3/MedDINOv3 segmentation experiments.

| Line | Class | Purpose | Base |
| --- | --- | --- | --- |
| 83 | `dinov3Trainer` | Runs nnU-Net training with DINOv3 backbone. | `nnUNetTrainer` |
| 1393 | `dinov3_base_sam_Trainer` | Runs nnU-Net training with DINOv3 backbone, base ViT, SAM-style decoder. | `dinov3Trainer` |
| 1432 | `dinov3_base_primus_Trainer` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder. | `dinov3Trainer` |
| 1521 | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, 3D context, augmentation without flips. | `dinov3_base_primus_Trainer` |
| 1713 | `dinov3_base_primus_Trainer_thin_3d_aug_flip` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, 3D context, custom augmentation. | `dinov3_base_primus_Trainer` |
| 1898 | `dinov3_base_primus_Trainer_2d_aug` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, 2D slices, custom augmentation. | `dinov3_base_primus_Trainer` |
| 2075 | `dinov3_base_primus_Trainer_scratch` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, training from scratch. | `dinov3_base_primus_Trainer` |
| 2112 | `dinov3_base_primus_Trainer_freeze` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 2158 | `dinov3_base_primus_Trainer_freeze_norm_meddinov3` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, MedDINOv3 CT normalization. | `dinov3_base_primus_Trainer` |
| 2205 | `dinov3_base_primus_Trainer_freeze_norm_window` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, CT window normalization. | `dinov3_base_primus_Trainer` |
| 2251 | `dinov3_base_primus_Trainer_freeze_norm_multi_window` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, CT window normalization. | `dinov3_base_primus_Trainer` |
| 2299 | `dinov3_base_primus_Trainer_freeze_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 2348 | `segdinov3_base_primus_Trainer` | Runs nnU-Net training with SegDINOv3 backbone, base ViT, Primus decoder. | `dinov3_base_primus_Trainer` |
| 2395 | `segdinov3_base_primus_Trainer_freeze` | Runs nnU-Net training with SegDINOv3 backbone, base ViT, Primus decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 2443 | `dinov3_base_primus_Trainer_freeze3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 2489 | `dinov3_small_primus_Trainer_freeze3D` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 2535 | `dinov3_base_primus_Trainer_Adapter3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, adapter tuning, 3D context. | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` |
| 2567 | `dinov3_small_primus_Trainer_Adapter3D_multi` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, adapter tuning, 3D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2600 | `dinov3_base_primus_Trainer_Adapter3D_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, adapter tuning, 3D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2634 | `dinov3_base_primus_Trainer_Adapter2_5D_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, adapter tuning, 2.5D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2666 | `dinov3_base_primus_Trainer_Lora3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 3D context. | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` |
| 2707 | `dinov3_small_primus_Trainer_Lora3D_multi` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, LoRA tuning, 3D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2749 | `dinov3_small_primus_Trainer_Lora3D_multi_256` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, LoRA tuning, 3D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2790 | `dinov3_base_primus_Trainer_Lora3D_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 3D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2830 | `dinov3_base_primus_Trainer_Lora2D_multi_debug` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 2D slices. | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` |
| 2871 | `dinov3_base_primus_Trainer_Lora2_5D_multi_debug` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 2.5D context. | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` |
| 2917 | `dinov3_base_primus_Trainer_Lora2_5D_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 2.5D context. | `dinov3_base_primus_Trainer_2d_aug` |
| 2960 | `meddinov3_base_primus_Trainer_Lora3D_multi` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, LoRA tuning, 3D context. | `dinov3_base_primus_Trainer` |
| 3010 | `dinov3_base_primus_Trainer_freeze3D_multi` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_freeze3D` |
| 3039 | `dinov3_small_primus_Trainer_freeze3D_multi` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_freeze3D` |
| 3070 | `dinov3_base_primus_Trainer_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_freeze3D` |
| 3098 | `dinov3_small_primus_Trainer_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_freeze3D` |
| 3125 | `dinov3_base_primus_Trainer_freeze3D_320` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_freeze3D` |
| 3155 | `dinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_base_primus_Trainer` |
| 3217 | `dinov3_plane_cycle_base_primus_Trainer_linear_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` |
| 3247 | `dinov3_plane_cycle_base_primus_Trainer_linear_freeze3D_320` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` |
| 3278 | `dinov3_plane_cycle_base_primus_Trainer_freeze3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_base_primus_Trainer` |
| 3341 | `dinov3_plane_cycle_base_primus_multiscale_Trainer_freeze3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_plane_cycle_base_primus_Trainer_freeze3D` |
| 3391 | `dinov3_plane_cycle_base_primus_multiscale_Trainer_Lora3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, LoRA tuning, PlaneCycle 3D encoder, 3D context. | `dinov3_base_primus_Trainer_thin_3d_aug_noflip` |
| 3453 | `dinov3_plane_cycle_base_primus_Trainer_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_plane_cycle_base_primus_Trainer_freeze3D` |
| 3485 | `dinov3_plane_cycle_base_primus_Trainer_freeze3D_320` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_plane_cycle_base_primus_Trainer_freeze3D` |
| 3516 | `dinov3_base_primus_Trainer_linear_freeze` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 3564 | `dinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer2` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 2 features. | `dinov3_base_primus_Trainer` |
| 3611 | `meddinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer2` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 2 features. | `dinov3_base_primus_Trainer` |
| 3663 | `meddinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer5` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 5 features. | `dinov3_base_primus_Trainer` |
| 3715 | `meddinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer8` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 8 features. | `dinov3_base_primus_Trainer` |
| 3767 | `meddinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer11` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 11 features. | `dinov3_base_primus_Trainer` |
| 3819 | `dinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer5` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 5 features. | `dinov3_base_primus_Trainer` |
| 3867 | `dinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer8` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 8 features. | `dinov3_base_primus_Trainer` |
| 3914 | `dinov3_base_primus_Trainer_linear_freeze_free_tokenizer_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, trainable tokenizer, DINO block 11 features. | `dinov3_base_primus_Trainer` |
| 3963 | `dinov3_base_primus_Trainer_linear_freeze_layer2` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 2 features. | `dinov3_base_primus_Trainer` |
| 4011 | `dinov3_base_primus_Trainer_linear_freeze_layer5` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 5 features. | `dinov3_base_primus_Trainer` |
| 4059 | `dinov3_base_primus_Trainer_linear_freeze_layer8` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 8 features. | `dinov3_base_primus_Trainer` |
| 4107 | `dinov3_base_primus_Trainer_linear_freeze_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer` |
| 4237 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E1_layernorm` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4253 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E2_lora_qkv_all_blocks` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, LoRA tuning, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4269 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E3_lora_qkv_middle_best4` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, LoRA tuning, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4285 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E4_lora_mlp_all_blocks` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, LoRA tuning, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4301 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E5_unfreeze_last4` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4317 | `dinov3_base_primus_Trainer_linear_freeze_layer11_E6_unfreeze_middle_best4` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer_linear_freeze_layer11` |
| 4333 | `dinov3_base_primus_Trainer_linear_freeze_norm_type_meddinov3` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, MedDINOv3 CT normalization. | `dinov3_base_primus_Trainer` |
| 4380 | `dinov3_base_primus_Trainer_linear_freeze_norm_type_window` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, CT window normalization. | `dinov3_base_primus_Trainer` |
| 4428 | `dinov3_base_primus_Trainer_linear_freeze_norm_type_multi_window` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, CT window normalization. | `dinov3_base_primus_Trainer` |
| 4476 | `dinov3_base_primus_Trainer_linear_freeze_256` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `dinov3_base_primus_Trainer_linear_freeze` |
| 4504 | `dinov3_base_primus_Trainer_linear_freeze_1024` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `dinov3_base_primus_Trainer_linear_freeze` |
| 4533 | `dinov3_base_primus_Trainer_linear_freeze_2048` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `dinov3_base_primus_Trainer_linear_freeze` |
| 4563 | `dinov3_base_primus_Trainer_linear_freeze3D` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 4611 | `dinov3_small_primus_Trainer_linear_freeze3D` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 4659 | `dinov3_base_primus_Trainer_linear_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_linear_freeze3D` |
| 4689 | `dinov3_small_primus_Trainer_linear_freeze3D_256` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_linear_freeze3D` |
| 4719 | `dinov3_base_primus_Trainer_linear_freeze3D_320` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer_linear_freeze3D` |
| 4749 | `dinov3_base_primus_multiscale_Trainer` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion. | `dinov3_base_primus_Trainer` |
| 4795 | `dinov3_base_primus_multiscale_Trainer_Lora` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, LoRA tuning. | `dinov3_base_primus_Trainer` |
| 4852 | `dinov3_base_primus_multiscale_Trainer_Adapter` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, adapter tuning. | `dinov3_base_primus_Trainer` |
| 4900 | `dinov3_small_primus_multiscale_Trainer` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, multi-scale feature fusion. | `dinov3_base_primus_Trainer` |
| 4957 | `dinov3_small_primus_multiscale_Trainer_Lora` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, multi-scale feature fusion, LoRA tuning. | `dinov3_base_primus_Trainer` |
| 5015 | `meddinov3_base_primus_multiscale_Trainer` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion. | `dinov3_base_primus_Trainer` |
| 5067 | `meddinov3_base_primus_multiscale_Trainer_Lora` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, LoRA tuning. | `dinov3_base_primus_Trainer` |
| 5133 | `meddinov3_base_primus_Trainer_freeze` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 5185 | `meddinov3_base_primus_Trainer_freeze_multi` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 5239 | `meddinov3_base_primus_Trainer_freeze3D` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 5291 | `meddinov3_base_primus_Trainer_freeze3D_256` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `meddinov3_base_primus_Trainer_freeze3D` |
| 5325 | `meddinov3_base_primus_Trainer_freeze3D_320` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, 3D context. | `meddinov3_base_primus_Trainer_freeze3D` |
| 5360 | `meddinov3_plane_cycle_base_primus_Trainer_freeze3D` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_base_primus_Trainer` |
| 5424 | `meddinov3_plane_cycle_base_primus_Trainer_freeze3D_256` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `meddinov3_plane_cycle_base_primus_Trainer_freeze3D` |
| 5470 | `meddinov3_plane_cycle_base_primus_Trainer_freeze3D_320` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `meddinov3_plane_cycle_base_primus_Trainer_freeze3D` |
| 5520 | `meddinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `dinov3_base_primus_Trainer` |
| 5592 | `meddinov3_plane_cycle_base_primus_Trainer_linear_freeze3D_256` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `meddinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` |
| 5645 | `meddinov3_plane_cycle_base_primus_Trainer_linear_freeze3D_320` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, PlaneCycle 3D encoder, 3D context. | `meddinov3_plane_cycle_base_primus_Trainer_linear_freeze3D` |
| 5700 | `meddinov3_base_primus_Trainer_linear_freeze` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `dinov3_base_primus_Trainer` |
| 5755 | `meddinov3_base_primus_Trainer_linear_freeze_layer2` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 2 features. | `dinov3_base_primus_Trainer` |
| 5811 | `meddinov3_base_primus_Trainer_linear_freeze_layer5` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 5 features. | `dinov3_base_primus_Trainer` |
| 5866 | `meddinov3_base_primus_Trainer_linear_freeze_layer8` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 8 features. | `dinov3_base_primus_Trainer` |
| 5921 | `meddinov3_base_primus_Trainer_linear_freeze_layer11` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, DINO block 11 features. | `dinov3_base_primus_Trainer` |
| 5976 | `meddinov3_base_primus_Trainer_linear_freeze_256` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `meddinov3_base_primus_Trainer_linear_freeze` |
| 6010 | `meddinov3_base_primus_Trainer_linear_freeze_320` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `meddinov3_base_primus_Trainer_linear_freeze` |
| 6044 | `meddinov3_base_primus_Trainer_linear_freeze_1024` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `meddinov3_base_primus_Trainer_linear_freeze` |
| 6080 | `meddinov3_base_primus_Trainer_linear_freeze_2048` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone. | `meddinov3_base_primus_Trainer_linear_freeze` |
| 6116 | `meddinov3_base_primus_Trainer_linear_freeze3D` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `dinov3_base_primus_Trainer` |
| 6170 | `meddinov3_base_primus_Trainer_linear_freeze3D_256` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `meddinov3_base_primus_Trainer_linear_freeze3D` |
| 6203 | `meddinov3_base_primus_Trainer_linear_freeze3D_320` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, Primus decoder, linear decoder, frozen backbone, 3D context. | `meddinov3_base_primus_Trainer_linear_freeze3D` |

| Line | Helper |
| --- | --- |
| 300 | `build_network_architecture` |
| 376 | `_build_loss` |
| 1410 | `build_network_architecture` |
| 1452 | `build_network_architecture` |
| 2095 | `build_network_architecture` |
| 2133 | `build_network_architecture` |
| 2179 | `build_network_architecture` |
| 2226 | `build_network_architecture` |
| 2272 | `build_network_architecture` |
| 2320 | `build_network_architecture` |
| 2369 | `build_network_architecture` |
| 2416 | `build_network_architecture` |
| 2464 | `build_network_architecture` |
| 2510 | `build_network_architecture` |
| 2539 | `build_network_architecture` |
| 2571 | `build_network_architecture` |
| 2604 | `build_network_architecture` |
| 2638 | `build_network_architecture` |
| 2670 | `build_network_architecture` |
| 2711 | `build_network_architecture` |
| 2753 | `build_network_architecture` |
| 2794 | `build_network_architecture` |
| 2834 | `build_network_architecture` |
| 2875 | `build_network_architecture` |
| 2921 | `build_network_architecture` |
| 2964 | `build_network_architecture` |
| 3014 | `build_network_architecture` |
| 3043 | `build_network_architecture` |
| 3073 | `build_network_architecture` |
| 3101 | `build_network_architecture` |
| 3128 | `build_network_architecture` |
| 3176 | `build_network_architecture` |
| 3221 | `build_network_architecture` |
| 3251 | `build_network_architecture` |
| 3299 | `build_network_architecture` |
| 3361 | `build_network_architecture` |
| 3411 | `build_network_architecture` |
| 3456 | `build_network_architecture` |
| 3488 | `build_network_architecture` |
| 3537 | `build_network_architecture` |
| 3585 | `build_network_architecture` |
| 3632 | `build_network_architecture` |
| 3684 | `build_network_architecture` |
| 3736 | `build_network_architecture` |
| 3788 | `build_network_architecture` |
| 3840 | `build_network_architecture` |
| 3888 | `build_network_architecture` |
| 3935 | `build_network_architecture` |
| 3984 | `build_network_architecture` |
| 4032 | `build_network_architecture` |
| 4080 | `build_network_architecture` |
| 4128 | `build_network_architecture` |
| 4170 | `_build_layer11_primus_linear` |
| 4241 | `build_network_architecture` |
| 4257 | `build_network_architecture` |
| 4273 | `build_network_architecture` |
| 4289 | `build_network_architecture` |
| 4305 | `build_network_architecture` |
| 4321 | `build_network_architecture` |
| 4354 | `build_network_architecture` |
| 4401 | `build_network_architecture` |
| 4449 | `build_network_architecture` |
| 4479 | `build_network_architecture` |
| 4507 | `build_network_architecture` |
| 4536 | `build_network_architecture` |
| 4584 | `build_network_architecture` |
| 4632 | `build_network_architecture` |
| 4662 | `build_network_architecture` |
| 4692 | `build_network_architecture` |
| 4722 | `build_network_architecture` |
| 4769 | `build_network_architecture` |
| 4815 | `build_network_architecture` |
| 4872 | `build_network_architecture` |
| 4920 | `build_network_architecture` |
| 4977 | `build_network_architecture` |
| 5035 | `build_network_architecture` |
| 5087 | `build_network_architecture` |
| 5154 | `build_network_architecture` |
| 5206 | `build_network_architecture` |
| 5260 | `build_network_architecture` |
| 5294 | `build_network_architecture` |
| 5328 | `build_network_architecture` |
| 5381 | `build_network_architecture` |
| 5428 | `build_network_architecture` |
| 5474 | `build_network_architecture` |
| 5541 | `build_network_architecture` |
| 5596 | `build_network_architecture` |
| 5649 | `build_network_architecture` |
| 5722 | `build_network_architecture` |
| 5777 | `build_network_architecture` |
| 5833 | `build_network_architecture` |
| 5888 | `build_network_architecture` |
| 5943 | `build_network_architecture` |
| 5979 | `build_network_architecture` |
| 6013 | `build_network_architecture` |
| 6047 | `build_network_architecture` |
| 6083 | `build_network_architecture` |
| 6139 | `build_network_architecture` |
| 6173 | `build_network_architecture` |
| 6206 | `build_network_architecture` |
| 6237 | `build_dinov3_base` |
| 6275 | `_build_sam` |

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinomed2dFintuner.py`

2D finetuning and pretraining trainer variants, including small/base encoder and prompt-loss experiments.

| Line | Class | Purpose | Base |
| --- | --- | --- | --- |
| 93 | `DinomedPromptLossBase` | Computes an auxiliary training loss. | `nn.Module` |
| 148 | `DinomedPromptLossV0` | Computes an auxiliary training loss. | `DinomedPromptLossBase` |
| 166 | `DinomedPromptLossV1` | Computes an auxiliary training loss. | `DinomedPromptLossBase` |
| 190 | `DinomedPromptLossV2` | Computes an auxiliary training loss. | `DinomedPromptLossBase` |
| 218 | `DinomedPromptLossV3` | Computes an auxiliary training loss. | `DinomedPromptLossBase` |
| 244 | `DinomedLossWithDatasetIds` | Computes an auxiliary training loss. | `nn.Module` |
| 256 | `dinomedFintuner` | Runs nnU-Net training with MedDINOv3 backbone. | `nnUNetTrainer` |
| 1566 | `dinov3_base_encoder_primus_decoder_Trainer` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder. | `dinomedFintuner` |
| 1656 | `_dinov3_base_primus_Trainer_linear_free0_x_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, DINO block 11 features. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 1719 | `dinov3_base_primus_Trainer_linear_free0_2_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, DINO block 11 features. | `_dinov3_base_primus_Trainer_linear_free0_x_layer11` |
| 1724 | `dinov3_base_primus_Trainer_linear_free0_5_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, DINO block 11 features. | `_dinov3_base_primus_Trainer_linear_free0_x_layer11` |
| 1729 | `dinov3_base_primus_Trainer_linear_free0_8_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, DINO block 11 features. | `_dinov3_base_primus_Trainer_linear_free0_x_layer11` |
| 1734 | `dinov3_base_primus_Trainer_linear_free0_11_layer11` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, linear decoder, DINO block 11 features. | `_dinov3_base_primus_Trainer_linear_free0_x_layer11` |
| 1739 | `dinov3_base_encoder_primus_decoder_Trainer_thin_3d_aug_noflip` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, 3D context, augmentation without flips. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 1931 | `dinov3_base_encoder_primus_decoder_Trainer_2d_aug_noflip` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, 2D slices, augmentation without flips. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2107 | `dinov3_small_encoder_multiscale_freeze_primus_decoder_Trainer` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, multi-scale feature fusion, frozen backbone. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2154 | `dinov3_small_encoder_multiscale_adapter_primus_decoder_Trainer` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, multi-scale feature fusion, adapter tuning. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2201 | `dinov3_small_encoder_multiscale_adapter_freeze_primus_decoder_Trainer` | Runs nnU-Net training with DINOv3 backbone, small ViT, Primus decoder, multi-scale feature fusion, adapter tuning, frozen backbone. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2248 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_seg_encoderonly_pretrained` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2296 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_seg_decoder_pretrained` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_seg_encoderonly_pretrained` |
| 2301 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_infonce_encoderonly_pretrained` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective, InfoNCE loss. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_seg_encoderonly_pretrained` |
| 2306 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_clipv0_encoderonly_pretrained` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective, CLIP-style contrastive loss. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_ours_seg_encoderonly_pretrained` |
| 2314 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_clip_v0_pretrained` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective, InfoNCE loss, CLIP-style contrastive loss. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2361 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_pretrained_v1` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective, InfoNCE loss. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_clip_v0_pretrained` |
| 2365 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_pretrained_v2` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, pretraining objective, InfoNCE loss. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_clip_v0_pretrained` |
| 2371 | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_dsc_supervised_nnunet` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, adapter tuning, frozen backbone, Dice/CE supervision. | `dinov3_small_encoder_multiscale_adapter_freeze_linear_decoder_Trainer_infonce_clip_v0_pretrained` |
| 2378 | `dinov3_small_encoder_multiscale_lora_freeze_linear_decoder_Trainer_dsc_supervised` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, LoRA tuning, frozen backbone, Dice/CE supervision. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2439 | `dinov3_small_encoder_multiscale_freeze_linear_decoder_Trainer` | Runs nnU-Net training with DINOv3 backbone, small ViT, multi-scale feature fusion, linear decoder, frozen backbone. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2488 | `dinov3_base_primus_multiscale_Trainer_Adapter` | Runs nnU-Net training with DINOv3 backbone, base ViT, Primus decoder, multi-scale feature fusion, adapter tuning. | `dinov3_base_encoder_primus_decoder_Trainer` |
| 2540 | `dinomed_base_PreTrainer_2D` | Runs nnU-Net training with MedDINOv3 backbone, base ViT, 2D slices, pretraining objective. | `dinomedFintuner` |
| 3123 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective. | `dinomed_base_PreTrainer_2D` |
| 3188 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_CosineMean` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3196 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_InfoNCE` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, InfoNCE loss. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3204 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_DSC_CE` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, Dice/CE supervision. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3212 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_InfoNCE_CLIP_V0` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, InfoNCE loss, CLIP-style contrastive loss. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3220 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_InfoNCE_CLIP_V1` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, InfoNCE loss, CLIP-style contrastive loss. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3228 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_InfoNCE_CLIP_V2` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, InfoNCE loss, CLIP-style contrastive loss. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |
| 3236 | `dinomed_small_adapter_PreTrainer_2D_Encoder_only_InfoNCE_CLIP_V3` | Runs nnU-Net training with MedDINOv3 backbone, small ViT, adapter tuning, 2D slices, pretraining objective, InfoNCE loss, CLIP-style contrastive loss. | `dinomed_small_adapter_PreTrainer_2D_Encoder_only` |

| Line | Helper |
| --- | --- |
| 473 | `build_network_architecture` |
| 549 | `_build_loss` |
| 1586 | `build_network_architecture` |
| 1676 | `build_network_architecture` |
| 2127 | `build_network_architecture` |
| 2174 | `build_network_architecture` |
| 2221 | `build_network_architecture` |
| 2268 | `build_network_architecture` |
| 2334 | `build_network_architecture` |
| 2398 | `build_network_architecture` |
| 2459 | `build_network_architecture` |
| 2508 | `build_network_architecture` |
| 2566 | `build_network_architecture` |
| 2839 | `_build_loss` |
| 3146 | `build_network_architecture` |

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinomed3dFintuner.py`

Legacy/disabled 3D finetuning scratchpad; currently mostly commented after removing dinov3_video experiments.

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py`

Primus decoder/adaptor model variants used by many DINO trainer classes.

| Line | Class | Purpose | Base |
| --- | --- | --- | --- |
| 228 | `NATFeatureFusion` | Pure PyTorch replacement for NATFeatureFusion. DINO-guided local cross-attention: - Query from DINO - Key/Value from CNN local neighborhoods - Residual correction added back to DINO Input: dino_feat: [B, C, H, W] cnn_feat : [B, C, H, W] Output: fused_feat: [B, C, H, W] | `nn.Module` |
| 362 | `PatchDecodeTrilinear` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 400 | `PatchDecodeBilinear` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 413 | `PrimusLinear3D` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 498 | `PrimusLinear` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 630 | `PatchDecode` | Loosely inspired by SAM decoder https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53 | `nn.Module` |
| 682 | `PatchDecode2_5D` | Stable decoder for shallow 3D feature: [B, C, D, H, W] where D is shallow (1~16), H/W large. Strategy: - early stages: only XY upsample - feature channels //2 each stage - light XY smoothing - final Z-aware refinement | `nn.Module` |
| 768 | `PatchDecode3D` | Loosely inspired by SAM decoder https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53 | `nn.Module` |
| 837 | `Decoder` | Builds an experimental model component used by DINO/MedDINO trainers. | `nn.Module` |
| 842 | `Primus` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 894 | `Primus3D` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 983 | `Primus3D_cycle` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1060 | `Primus3D_Multiscale_cycle` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1142 | `Primus3D_linear_cycle` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1220 | `Primus_Multiscale` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1276 | `DPT_Multiscale` | Experimental helper class used by the local DINO/MedDINO code. | `AbstractDynamicNetworkArchitectures` |
| 1331 | `Primus_Multiscale3D` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1445 | `Primus_Multiscale2D_debug` | Builds an experimental model component used by DINO/MedDINO trainers. | `Primus_Multiscale` |
| 1468 | `Primus_Multiscale2_5D` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1585 | `Primus_Multiscale2_5D_debug` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1702 | `MultiScale` | Experimental helper class used by the local DINO/MedDINO code. | `AbstractDynamicNetworkArchitectures` |
| 1900 | `PrimusLinearUNetAdapterNaT` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/multi_layer_fusion.py`

Multi-layer feature fusion model variants.

| Line | Class | Purpose | Base |
| --- | --- | --- | --- |
| 20 | `ChannelGate3D` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 36 | `ChannelGate` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 51 | `AlignBlock` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 77 | `AlignBlockSimple` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 113 | `AlignBlockDepthSep` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 169 | `AlignBlock3DSimple` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 194 | `AlignBlock3DDepthSep` | Experimental helper class used by the local DINO/MedDINO code. | `nn.Module` |
| 249 | `Primus_Multiscale` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 295 | `MultiScale` | Experimental helper class used by the local DINO/MedDINO code. | `AbstractDynamicNetworkArchitectures` |
| 341 | `MultiLayerAlignAndFusionBaseline0` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 501 | `MultiLayerAlignAndFusionBaseline1` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 929 | `MultiLayerAlignAndFusion3DBaseline0` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1098 | `MultiLayerAlignAndFusion3DBaseline01` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 1297 | `MultiLayerAlignAndFusion3DBaseline1` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |

## `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/dinomed3d_pretrain.py`

2D encoder/decoder wrappers used by pretraining/finetuning experiments.

| Line | Class | Purpose | Base |
| --- | --- | --- | --- |
| 19 | `Decoder` | Builds an experimental model component used by DINO/MedDINO trainers. | `nn.Module` |
| 24 | `PatchDecode` | Loosely inspired by SAM decoder https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53 | `nn.Module` |
| 76 | `Multiscale_2d_encoder_only` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |
| 169 | `Multiscale_2d_conv_decoder` | Builds an experimental model component used by DINO/MedDINO trainers. | `AbstractDynamicNetworkArchitectures` |

