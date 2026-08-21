import torch
import inspect
import multiprocessing
import os
import shutil
import sys
import copy


# navigate relative to this file
dino_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3")

if dino_root not in sys.path:
    sys.path.append(dino_root)

import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
# from torch import GradScaler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, CosineSimilarityandL1ToMean, CosineSimilarityandL1ToMeanDist, InstanceSpatialContrastiveCompactLoss2D
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch.nn.functional as F
import random
from nnunetv2.training.nnUNetTrainer.udinov3Trainer import udinov3_base_primus_Trainer

import copy 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from sklearn.manifold import TSNE


def draw_tsne_points_colored_by_instance_3D(
    output_np,
    target_np,
    save_folder,
    batch_id=0,
    ignore_ids=(0,),
    max_points_per_instance=200,
    min_voxels_per_instance=10,
    max_total_points=5000,
    normalize_feature=False,
    annotate_instances=True,
    point_size=8,
    alpha=0.7,
    random_state=42,
):
    """
    Draw voxel-level t-SNE for a 3D feature volume, colored by instance id.

    Args:
        output_np: np.ndarray, shape (C, H, W, D)
            Feature volume.
        target_np: np.ndarray, shape (1, H, W, D) or (H, W, D)
            Instance id map. Each non-zero value is one instance id.
        save_folder: str
            Folder to save the figure.
        batch_id: int
            Batch index for naming the output file.
        ignore_ids: tuple[int]
            Instance ids to ignore, typically background (0,).
        max_points_per_instance: int
            Maximum sampled voxels per instance.
        min_voxels_per_instance: int
            Minimum voxel count required for an instance to be included.
        max_total_points: int
            Additional global cap on total points for t-SNE speed/stability.
        normalize_feature: bool
            Whether to L2-normalize each voxel feature before t-SNE.
        annotate_instances: bool
            Whether to put instance id text at each cluster center.
        point_size: int
            Scatter point size.
        alpha: float
            Scatter transparency.
        random_state: int
            Random seed.
    """

    print("output_np", output_np.shape, "target_np", target_np.shape)

    # ---------------- sanity checks ----------------
    if output_np.ndim != 4:
        raise ValueError(f"Expected output_np shape (C,H,W,D), got {output_np.shape}")

    if target_np.ndim == 4:
        if target_np.shape[0] != 1:
            raise ValueError(f"Expected target_np shape (1,H,W,D) or (H,W,D), got {target_np.shape}")
        inst_map = target_np[0]
    elif target_np.ndim == 3:
        inst_map = target_np
    else:
        raise ValueError(f"Expected target_np shape (1,H,W,D) or (H,W,D), got {target_np.shape}")

    C, H, W, D = output_np.shape

    if inst_map.shape != (H, W, D):
        raise ValueError(
            f"Shape mismatch: output_np spatial shape is {(H, W, D)}, "
            f"but target_np spatial shape is {inst_map.shape}"
        )

    os.makedirs(save_folder, exist_ok=True)

    # ---------------- flatten ----------------
    # feat_flat: (N, C), inst_flat: (N,)
    feat_flat = output_np.reshape(C, -1).T
    inst_flat = inst_map.reshape(-1)

    unique_ids = np.unique(inst_flat)

    sampled_features = []
    sampled_instance_ids = []

    rng = np.random.default_rng(random_state)

    # ---------------- per-instance sampling ----------------
    for inst_id in unique_ids:
        if inst_id in ignore_ids:
            continue

        idx = np.where(inst_flat == inst_id)[0]
        if len(idx) < min_voxels_per_instance:
            continue

        if len(idx) > max_points_per_instance:
            idx = rng.choice(idx, size=max_points_per_instance, replace=False)

        sampled_features.append(feat_flat[idx])                 # (n_i, C)
        sampled_instance_ids.append(np.full(len(idx), inst_id)) # (n_i,)

    if len(sampled_features) == 0:
        print("No valid instance voxels for 3D t-SNE.")
        return

    sampled_features = np.concatenate(sampled_features, axis=0)
    sampled_instance_ids = np.concatenate(sampled_instance_ids, axis=0)

    if sampled_features.shape[0] < 2:
        print("Not enough points for 3D t-SNE.")
        return

    # ---------------- optional global downsample ----------------
    if sampled_features.shape[0] > max_total_points:
        keep_idx = rng.choice(sampled_features.shape[0], size=max_total_points, replace=False)
        sampled_features = sampled_features[keep_idx]
        sampled_instance_ids = sampled_instance_ids[keep_idx]

    # ---------------- optional feature normalization ----------------
    if normalize_feature:
        sampled_features = sampled_features / (
            np.linalg.norm(sampled_features, axis=1, keepdims=True) + 1e-8
        )

    # ---------------- t-SNE ----------------
    n_points = sampled_features.shape[0]
    perplexity = min(30, max(5, n_points // 100))
    perplexity = min(perplexity, n_points - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    emb_2d = tsne.fit_transform(sampled_features)

    # ---------------- color mapping by instance ----------------
    uniq_inst = np.unique(sampled_instance_ids)
    inst_to_color = {inst_id: i for i, inst_id in enumerate(uniq_inst)}
    color_idx = np.array([inst_to_color[x] for x in sampled_instance_ids])

    # ---------------- draw ----------------
    plt.figure(figsize=(8, 7))
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=color_idx,
        s=point_size,
        cmap="nipy_spectral",
        alpha=alpha,
        linewidths=0,
    )

    if annotate_instances:
        for inst_id in uniq_inst:
            mask = sampled_instance_ids == inst_id
            if mask.sum() == 0:
                continue
            center = emb_2d[mask].mean(axis=0)
            plt.text(
                center[0],
                center[1],
                str(int(inst_id)),
                fontsize=8,
                alpha=0.9,
                ha="center",
                va="center",
            )

    plt.title(f"3D voxel t-SNE colored by instance (batch {batch_id})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    save_path = join(
        save_folder,
        f"tsne_points_by_instance_3D_batch_{str(batch_id).zfill(4)}.png"
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved 3D t-SNE to:", save_path)



class udinov3_base_Linear_Prob_after_PreTrainer_3D_version0(udinov3_base_primus_Trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # Call the parent class constructor
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Override hyperparameters
        self.initial_lr = 3e-3
        self.vit_lr = 1e-4
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.warmup_epochs = 0
        self.num_epochs = 1000
        # self.save_every = 10
        self.current_epoch = 0
        self.enable_deep_supervision = False
        # self.supervise_rescale = 1/16

    @staticmethod
    def build_network_architecture(patch_size: tuple, 
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
    
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base
        # Initialize model
        model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05)
        # Load checkpoint
        chkpt = torch.load(
            '/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            map_location='cpu'
        )
        # Load with strict=False so it won’t crash on mismatches
        missing, unexpected = model.load_state_dict(chkpt, strict=False)
        print("Load dinov3 model", 'missing', missing, 'unexpected', unexpected)

        # from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PrimusLinearUNetAdapter
        # primus = PrimusLinearUNetAdapter(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
        #                            dino_encoder=model, freeze_backbone=True, feature_extraction=False)

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.multi_layer_fusion import MultiLayerAlignAndFusion3DBaseline0
        primus = MultiLayerAlignAndFusion3DBaseline0(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
                                   dino_encoder=model, freeze_backbone=True, feature_extraction=False, linear_prob=True, freeze_adapter=True)
                                   # ,  interaction_indices=[2,5,8,11]

        return primus


class udinov3_base_Decoder_Prob_after_PreTrainer_3D_version0(udinov3_base_primus_Trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # Call the parent class constructor
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Override hyperparameters
        self.initial_lr = 3e-3
        self.vit_lr = 1e-4
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.warmup_epochs = 0
        self.num_epochs = 1000
        # self.save_every = 10
        self.current_epoch = 0
        self.enable_deep_supervision = False
        # self.supervise_rescale = 1/16

    @staticmethod
    def build_network_architecture(patch_size: tuple, 
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
    
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base
        # Initialize model
        model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05)
        # Load checkpoint
        chkpt = torch.load(
            '/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            map_location='cpu'
        )
        # Load with strict=False so it won’t crash on mismatches
        missing, unexpected = model.load_state_dict(chkpt, strict=False)
        print("Load dinov3 model", 'missing', missing, 'unexpected', unexpected)

        # from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PrimusLinearUNetAdapter
        # primus = PrimusLinearUNetAdapter(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
        #                            dino_encoder=model, freeze_backbone=True, feature_extraction=False)

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.multi_layer_fusion import MultiLayerAlignAndFusion3DBaseline0
        primus = MultiLayerAlignAndFusion3DBaseline0(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
                                   dino_encoder=model, freeze_backbone=True, feature_extraction=False, decoder_prob=True, freeze_adapter=False)
                                   # ,  interaction_indices=[2,5,8,11]

        return primus







class udinov3_base_PreTrainer_3D_version01_with_norm(udinov3_base_primus_Trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # Call the parent class constructor
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Override hyperparameters
        self.initial_lr = 3e-3
        self.vit_lr = 1e-4
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50
        self.warmup_epochs = 0
        self.num_epochs = 300
        # self.save_every = 10
        self.current_epoch = 0
        self.enable_deep_supervision = False
        self.supervise_rescale = 1/16
        self.diff_norm_weight=0.5

    @staticmethod
    def build_network_architecture(patch_size: tuple, 
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
    
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base
        # Initialize model
        model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05)
        # Load checkpoint
        chkpt = torch.load(
            '/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            map_location='cpu'
        )
        # Load with strict=False so it won’t crash on mismatches
        missing, unexpected = model.load_state_dict(chkpt, strict=False)
        print("Load dinov3 model", 'missing', missing, 'unexpected', unexpected)

        # from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PrimusLinearUNetAdapter
        # primus = PrimusLinearUNetAdapter(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
        #                            dino_encoder=model, freeze_backbone=True, feature_extraction=False)

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.multi_layer_fusion import MultiLayerAlignAndFusion3DBaseline01
        primus = MultiLayerAlignAndFusion3DBaseline01(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
                                   dino_encoder=model, freeze_backbone=True, feature_extraction=True, linear_prob=False)
                                   # ,  interaction_indices=[2,5,8,11]

        return primus
    

    def perform_actual_validation(self, save_probabilities: bool = False):
        print("TODO perform actually eval with matching")


    def validation_step(self, batch: dict, draw=False) -> dict:
        data = batch['data']
        target = batch['target']
        if self.supervise_rescale is not None:
            target=target.float()
            target = F.interpolate(
                        target,
                        scale_factor=(1, self.supervise_rescale,self.supervise_rescale) ,
                        mode="nearest",
                    )
            target = target.int()

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            if draw:
                self.given_validation_step_draw(data, target, output, batch['keys'])
            del data
            l = self.loss(output, target)

        tp_hard = []
        fp_hard = []
        fn_hard = []

        if isinstance(output, list):
            output = output[0]
        if isinstance(target, list):
            target = target[0]

        # do visualization of validation predictions for pca 
        current_epoch = self.current_epoch
        keys = batch['keys'] if 'keys' in batch else None
        save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d' % current_epoch)
        os.makedirs(save_folder, exist_ok=True)
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def volume_to_pca_frames(self, data, output, target, save_folder):
        """
        data: torch tensor of shape (B, C, D, H, W) or (B, C, H, W) if 2D
        output: torch tensor of shape (B, C, D, H, W) or (B, C, H, W) if 2D
        target: torch tensor of shape (B, 1, D, H, W) or (B, 1, H, W) if 2D
        """
        # convert output and target to numpy and move channels to last dimension
        # draw data torch.Size([12, 1, 512, 512])
        os.makedirs(save_folder, exist_ok=True)
        plt.close('all')
        batch_size = data.shape[0]
        data_dim = len(data.shape)

        def single_batch_to_pca_frames_3D(data_np, output_np, target_np, save_folder, batch_id=0):
            pca_components = 6

            # incremental_pca = IncrementalPCA(n_components=3)
            PCA_fitting = PCA(n_components=pca_components)
            org_H, org_W, org_Z = output_np.shape[1], output_np.shape[2], output_np.shape[3] 
            if org_H*org_W*org_Z > 10000:
                print("Too many voxels for PCA. Downsampling PCA visualization.")
                output_np = output_np[:,::2,::2,::2]
                target_np = target_np[:,::2,::2,::2]
                data_np = data_np[:,::2,::2,::2]
                data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min()+ 1e-8)

            
            new_H, new_W, new_Z = output_np.shape[1], output_np.shape[2], output_np.shape[3]

            output_flatten = output_np.reshape(output_np.shape[0], -1).T  # (H*W*D, C)
            PCA_fitting.fit(output_flatten)
            output_pca = PCA_fitting.transform(output_flatten)  # (H*W*D, pca_components)
            output_pca = output_pca.T.reshape(pca_components, new_H, new_W, new_Z)  # (pca_components, H, W, D)

            # draw results as this, first row is gt image, second row is 0-3 PCA components, third row is 4-6 PCA components
            # in each image we cover 0-10 frame, so it will be 3row x 10 colume images
            for frame_num in range(new_H):
                if frame_num % 4 == 0:
                    # create new figure every 10 frames
                    save_path = join(save_folder, f'pca_visualization_frame_{frame_num}_batch_{batch_id}.png')
                    if frame_num != 0:
                        print("Save to ", save_path)
                        plt.savefig(save_path)
                    fig, axes = plt.subplots(4, 4, figsize=(10, 6))

                colume_to_draw = frame_num % 4      
                axes[1, colume_to_draw].imshow(target_np[0, frame_num], cmap='gray')
                axes[1, colume_to_draw].set_title(f'GT Frame {frame_num}', fontsize=6)

                for j in range(2):
                    start_component = j*3
                    end_component = (j+1)*3
                    pca_ij = output_pca[start_component:end_component, frame_num]  # (3, W, D)
                    pca_ij = (pca_ij - pca_ij.min()) / (pca_ij.max() - pca_ij.min())  # normalize to 0-1
                    pca_ij = np.transpose(pca_ij, (1, 2, 0))  # (W, D, 3)
                    # colormap pca rgb to strongly differentiate the components
                    axes[j+2, colume_to_draw].imshow(pca_ij, cmap='turbo')
                    axes[j+2, colume_to_draw].set_title(f'PCA {start_component}-{end_component} Frame {frame_num}', fontsize=6)
                # add data as the first row
                data_ij = data_np[:, frame_num]  # (C, W, D)
                # data_ij = (data_ij - data_ij.min()) / (data_ij.max() - data_ij.min())  # normalize to 0-1
                data_ij = np.transpose(data_ij, (1, 2, 0))  # (W, D, C)
                axes[0, colume_to_draw].imshow(data_ij, cmap='gray')
                # plt.tight_layout()
                # turn off axis for all images
                for i in range(4):
                    for j in range(4):
                        axes[i, j].axis('off')
                # plt.savefig(save_path)

                draw_tsne_points_colored_by_instance_3D(
                    output_np = output_np,
                    target_np = target_np,
                    save_folder = save_folder,
                    batch_id = batch_id
                )


        def single_batch_to_pca_frames_2D(data_np, output_np, target_np, save_folder, batch_id=0):
            pca_components = 6
            # incremental_pca = IncrementalPCA(n_components=3)
            PCA_fitting = PCA(n_components=pca_components)
            org_H, org_W = output_np.shape[1], output_np.shape[2] 
            save_path = join(save_folder, f'pca_visualization_batch_{str(batch_id).zfill(4)}.png')

            # print("data_np", data_np.shape, "output_np", output_np.shape, 'target_np', target_np.shape)
            # data_np (1, 512, 512) output_np (16, 512, 512) target_np (1, 512, 512) 
            if org_H*org_W > 1024*1024:
                print("Too many voxels for PCA. Downsampling PCA visualization.")
                output_np = output_np[:,::4,::4]
                target_np = target_np[:,::4,::4]
                data_np = data_np[:,::4,::4]

                data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min()+ 1e-8)
            
            new_H, new_W = output_np.shape[1], output_np.shape[2]

            output_flatten = output_np.reshape(output_np.shape[0], -1).T  # (H*W*D, C)
            PCA_fitting.fit(output_flatten)
            output_pca = PCA_fitting.transform(output_flatten)  # (H*W*D, pca_components)
            output_pca = output_pca.T.reshape(pca_components, new_H, new_W)  # (pca_components, H, W, D)

            frame_num = 0
            plt.close('all')
            fig, axes = plt.subplots(1, 4)
            colume_to_draw = 0
            axes[1].imshow(target_np[0], cmap='gray')
            axes[1].set_title(f'GT Frame {frame_num}', fontsize=6)

            for j in range(2):
                start_component = j*3
                end_component = (j+1)*3
                pca_ij = output_pca[start_component:end_component]  # (3, W, D)
                pca_ij = (pca_ij - pca_ij.min()) / (pca_ij.max() - pca_ij.min())  # normalize to 0-1
                pca_ij = np.transpose(pca_ij, (1, 2, 0))  # (W, D, 3)
                # colormap pca rgb to strongly differentiate the components
                axes[j+2].imshow(pca_ij, cmap='turbo')
                axes[j+2].set_title(f'PCA {start_component}-{end_component} Frame {frame_num}', fontsize=6)

            data_ij = data_np  # (C, W, D)
            # data_ij = (data_ij - data_ij.min()) / (data_ij.max() - data_ij.min())  # normalize to 0-1
            data_ij = np.transpose(data_ij, (1, 2, 0))  # (W, D, C)
            axes[0].imshow(data_ij, cmap='gray')
            # plt.tight_layout()
            # turn off axis for all images
            for i in range(4):
                # for j in range(10):
                axes[i].axis('off')
            plt.savefig(save_path)


        for batch_i in range(batch_size):
            # print("data", data.shape, 'output', output.shape, 'target', target.shape)
            data_np = data.detach().cpu().numpy()[batch_i]
            output_np = output.detach().cpu().numpy()[batch_i]
            target_np = target.detach().cpu().numpy()[batch_i]
            if data_dim == 5:
                single_batch_to_pca_frames_3D(data_np, output_np, target_np, save_folder, batch_id=batch_i)
            elif data_dim == 4:
                single_batch_to_pca_frames_2D(data_np, output_np, target_np, save_folder, batch_id=batch_i)


    def given_validation_step_draw(self, data, target, output, keys) -> dict:

        if isinstance(output, list):
            #     output = output[0]
            # if isinstance(target, list):
            #     target = target[0]
            list_length = len(output)
            for j in range(list_length):
                current_epoch = self.current_epoch
                # keys = batch['keys'] if 'keys' in batch else None
                save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d_output_%03d' % (current_epoch, j))
 
                self.volume_to_pca_frames(data, output[j], target[j], save_folder)

        else:
            # do visualization of validation predictions for pca 
            current_epoch = self.current_epoch
            # keys = batch['keys'] if 'keys' in batch else None
            save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d' % current_epoch)
            # if data_dim == 4:
            self.volume_to_pca_frames(data, output, target, save_folder)
        

    # def validation_step_draw(self, batch: dict) -> dict:
    #     data = batch['data']
    #     target = batch['target']
    #     # 2D model and 3D model
    #     # take one batch only
    #     data = data[0:1]
    #     target = target[0:1]

    #     # data_dim = len(data.shape)

    #     data = data.to(self.device, non_blocking=True)
    #     if isinstance(target, list):
    #         target = [i.to(self.device, non_blocking=True) for i in target]
    #     else:
    #         target = target.to(self.device, non_blocking=True)

    #     with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
    #         # print("start drawing forwarding")
    #         with torch.no_grad():
    #             output = self.network(data)

    #     # print("prepare drawing")
    #     # map to cpu

    #     if isinstance(output, list):
    #         #     output = output[0]
    #         # if isinstance(target, list):
    #         #     target = target[0]
    #         list_length = len(output)
    #         for j in range(list_length):
    #             current_epoch = self.current_epoch
    #             keys = batch['keys'] if 'keys' in batch else None
    #             save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d_output_%03d' % (current_epoch, j))
 
    #             self.volume_to_pca_frames(data, output[j], target[j], save_folder)

    #     else:
    #         # do visualization of validation predictions for pca 
    #         current_epoch = self.current_epoch
    #         keys = batch['keys'] if 'keys' in batch else None
    #         save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d' % current_epoch)
    #         # if data_dim == 4:
    #         self.volume_to_pca_frames(data, output, target, save_folder)
        
    def run_training(self):
        self.on_train_start()

        val_data_0 = next(iter(self.dataloader_val))
        val_data_0 = copy.deepcopy(val_data_0)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
                # print("Start drawing")
                # self.validation_step_draw(val_data_0)
                self.validation_step(val_data_0, draw=True)

            self.on_epoch_end()


        self.on_train_end()

    def _build_loss(self):
        # debug this loss also
        loss = InstanceSpatialContrastiveCompactLoss2D(
            temperature = 0.2,
            compact_weight = 0.5,
            normalize = True,
            min_points = 4,
            min_group_points = 2,
        )
        # CosineSimilarityandL1ToMean(min_points=4, supervise_manner='3D')
        # CosineSimilarityandL1ToMeanDist()
        return loss



    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        # tp = np.sum(outputs_collated['tp_hard'], 0)
        # fp = np.sum(outputs_collated['fp_hard'], 0)
        # fn = np.sum(outputs_collated['fn_hard'], 0)
        # print("outputs_collated", outputs_collated)

        if self.is_ddp:
            world_size = dist.get_world_size()

            # tps = [None for _ in range(world_size)]
            # dist.all_gather_object(tps, tp)
            # tp = np.vstack([i[None] for i in tps]).sum(0)

            # fps = [None for _ in range(world_size)]
            # dist.all_gather_object(fps, fp)
            # fp = np.vstack([i[None] for i in fps]).sum(0)

            # fns = [None for _ in range(world_size)]
            # dist.all_gather_object(fns, fn)
            # fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        # global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        # mean_fg_dice = np.nanmean(global_dc_per_class)
        # self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        # self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)



    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        # self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
        #                                        self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['val_losses'][-1]
            self.print_to_log_file(f"Yayy! New best EMA val loss: {np.round(self._best_ema, decimals=4)}")
            # self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # if self.local_rank == 0:
        #     self.logger.plot_pretrain_progress(self.output_folder)

        self.current_epoch += 1


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        if self.supervise_rescale is not None:
            target = target.float()
            target = F.interpolate(
                        target,
                        scale_factor=(1, self.supervise_rescale,self.supervise_rescale) ,
                        mode="nearest",
                    )
            target = target.int()
            # print("Data shape", data.shape, 'target', target.shape, 'unique instance', target.unique())

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, offset_list = self.network(data, return_offset=True)
            # print("output", output.shape, 'target', target.shape)
            # del data
            offset_reg_loss = 0
            for offset_i in offset_list:
                offset_reg_loss += self.diff_norm_weight * offset_i.abs().mean()

            l = self.loss(output, target) + offset_reg_loss

        if l is None or not l.requires_grad:
            print("skip batch: invalid loss")
            return {'loss': np.zeros(()).astype(np.float32), 'offset_reg_loss': offset_reg_loss.detach().cpu().numpy()}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        # print(".detach().cpu().numpy()",l.detach().cpu().numpy().shape)
        return {'loss': l.detach().cpu().numpy(), 'offset_reg_loss': offset_reg_loss.detach().cpu().numpy()}


    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)








class udinov3_base_PreTrainer_3D_version11_with_norm(udinov3_base_PreTrainer_3D_version01_with_norm):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # Call the parent class constructor
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Override hyperparameters
        self.initial_lr = 3e-3
        self.vit_lr = 1e-4
        self.weight_decay = 5e-2
        self.vit_weight_decay = 5e-2
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50
        self.warmup_epochs = 0
        self.num_epochs = 300
        # self.save_every = 10
        self.current_epoch = 0
        self.enable_deep_supervision = False
        self.supervise_rescale = None
        self.diff_norm_weight = 1.0

    @staticmethod
    def build_network_architecture(patch_size: tuple, 
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
    
        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base
        # Initialize model
        model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05)
        # Load checkpoint
        chkpt = torch.load(
            '/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            map_location='cpu'
        )
        # Load with strict=False so it won’t crash on mismatches
        missing, unexpected = model.load_state_dict(chkpt, strict=False)
        print("Load dinov3 model", 'missing', missing, 'unexpected', unexpected)

        # from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import PrimusLinearUNetAdapter
        # primus = PrimusLinearUNetAdapter(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
        #                            dino_encoder=model, freeze_backbone=True, feature_extraction=False)

        from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.multi_layer_fusion import MultiLayerAlignAndFusion3DBaseline01
        primus = MultiLayerAlignAndFusion3DBaseline01(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
                                   dino_encoder=model, freeze_backbone=True, feature_extraction=True, linear_prob=False, pretrain_decoder=True)
                                   # ,  interaction_indices=[2,5,8,11]

        return primus
    


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        if self.supervise_rescale is not None:
            target = target.float()
            target = F.interpolate(
                        target,
                        scale_factor=(1, self.supervise_rescale,self.supervise_rescale) ,
                        mode="nearest",
                    )
            target = target.int()
            # print("Data shape", data.shape, 'target', target.shape, 'unique instance', target.unique())

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, offset_list = self.network(data, return_offset=True)
            # print("output", output.shape, 'target', target.shape)
            # del data
            offset_reg_loss = 0
            for offset_i in offset_list:
                offset_reg_loss += 0.1 * offset_i.abs().mean()

            l = self.loss(output, target) + offset_reg_loss

        if l is None or not l.requires_grad:
            print("skip batch: invalid loss")
            return {'loss': np.zeros(()).astype(np.float32), 'offset_reg_loss': offset_reg_loss.detach().cpu().numpy()}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        # print(".detach().cpu().numpy()",l.detach().cpu().numpy().shape)
        return {'loss': l.detach().cpu().numpy(), 'offset_reg_loss': offset_reg_loss.detach().cpu().numpy()}



    def validation_step(self, batch: dict, draw=False) -> dict:
        data = batch['data']
        target = batch['target']
        if self.supervise_rescale is not None:
            target=target.float()
            target = F.interpolate(
                        target,
                        scale_factor=(1, self.supervise_rescale,self.supervise_rescale) ,
                        mode="nearest",
                    )
            target = target.int()

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output , offset_list= self.network(data, return_offset=True)
            if draw:
                self.given_validation_step_draw(data, target, output, batch['keys'])
            del data
            l = self.loss(output, target)

            offset_reg_loss = 0
            for offset_i in offset_list:
                offset_reg_loss += 0.1 * offset_i.abs().mean()
            l = l + offset_reg_loss


        tp_hard = []
        fp_hard = []
        fn_hard = []

        if isinstance(output, list):
            output = output[0]
        if isinstance(target, list):
            target = target[0]

        # do visualization of validation predictions for pca 
        current_epoch = self.current_epoch
        keys = batch['keys'] if 'keys' in batch else None
        save_folder = join(self.output_folder, 'validation_visualization_epoch_%03d' % current_epoch)
        os.makedirs(save_folder, exist_ok=True)
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard, 'offset_reg_loss': offset_reg_loss.detach().cpu().numpy()}



    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        # tp = np.sum(outputs_collated['tp_hard'], 0)
        # fp = np.sum(outputs_collated['fp_hard'], 0)
        offset_reg_loss = np.sum(outputs_collated['offset_reg_loss'], 0)
        # print("outputs_collated", outputs_collated)

        if self.is_ddp:
            world_size = dist.get_world_size()


            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
            loss_norm_here = np.mean(outputs_collated['offset_reg_loss'])

        # global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        # mean_fg_dice = np.nanmean(global_dc_per_class)
        # self.logger.log('loss_norm_here', loss_norm_here, self.current_epoch)
        print("loss_norm_here", loss_norm_here, self.current_epoch)
        print("val_losses", loss_here, self.current_epoch)
        # self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)



    def _build_loss(self):
        # debug this loss also
        loss = InstanceSpatialContrastiveCompactLoss2D(
            temperature = 0.2,
            compact_weight = 0.5,
            normalize = True,
            min_points = 64,
            min_group_points = 8,
        )
        # CosineSimilarityandL1ToMean(min_points=4, supervise_manner='3D')
        # CosineSimilarityandL1ToMeanDist()
        return loss
