import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F


class CosineSimilarityandL1ToMeanDist(nn.Module):
    def __init__(self, ignore_classes=[0], min_points=25, inter_weight=0.01, threshold_distance=20):
        super().__init__()
        self.ignore_classes = ignore_classes
        self.min_points = min_points
        self.inter_weight = inter_weight
        self.threshold_distance = threshold_distance # less 20 voxel center will be fine
        # self.sigma = sigma

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # net_output: B,C,H,W,D
        # target: B,H,W,D

        B, C, H, W, D = net_output.shape
        device = net_output.device

        # ---------------- feature flatten ----------------
        features = net_output.view(B, C, -1)
        features = features.permute(0, 2, 1).contiguous().view(-1, C)

        labels = target.view(-1)

        # ---------------- coordinate grid ----------------
        zz, yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            torch.arange(D, device=device),
            indexing='ij'
        )

        coords = torch.stack([zz, yy, xx], dim=-1).view(-1, 3)   # (HWD, 3)
        coords = coords.repeat(B, 1)                             # (B*HWD, 3)

        unique_labels = torch.unique(labels)

        class_means = []
        class_centroids = []
        intra_losses = []

        for lab in unique_labels:
            if lab.item() in self.ignore_classes:
                continue

            indices = (labels == lab).nonzero(as_tuple=True)[0]

            if indices.numel() < self.min_points:
                continue

            class_features = features[indices]
            class_coords = coords[indices]

            # ---------------- feature mean ----------------
            class_mean = class_features.mean(dim=0, keepdim=True)

            sim_to_mean = F.cosine_similarity(class_features, class_mean, dim=1)
            intra_loss = 1.0 - sim_to_mean.mean()
            intra_losses.append(intra_loss)

            class_means.append(F.normalize(class_mean.squeeze(0), dim=0))

            # ---------------- spatial centroid ----------------
            class_centroid = class_coords.float().mean(dim=0)
            class_centroids.append(class_centroid)

        if len(class_means) == 0:
            # print("No class loss")
            return torch.tensor(0.0, device=device)

        intra_loss = torch.stack(intra_losses).mean()

        if len(class_means) < 2:
            # print("Less class loss")
            return intra_loss

        class_means = torch.stack(class_means, dim=0)         # K,C
        class_centroids = torch.stack(class_centroids, dim=0) # K,3

        # ---------------- cosine similarity ----------------
        sim_matrix = torch.matmul(class_means, class_means.T)

        # ---------------- centroid distance ----------------
        dist_matrix = torch.cdist(class_centroids, class_centroids, p=2)

        # ---------------- Gaussian weight ----------------
        # weight_matrix = torch.exp(-(dist_matrix ** 2) / (2 * self.sigma ** 2))
        weight_matrix = torch.clamp(dist_matrix / self.threshold_distance, 0.0, 1.0) ** 2

        eye = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)

        sim_vals = sim_matrix[~eye]
        weight_vals = weight_matrix[~eye]

        inter_loss = (sim_vals * weight_vals).sum() / (weight_vals.sum() + 1e-8)

        loss = intra_loss + self.inter_weight * inter_loss
        # print("intra_loss", intra_loss, 'inter_loss', inter_loss)
        return loss

class CosineSimilarityandL1ToMean(nn.Module):
    def __init__(self, ignore_classes=[0], min_points=25, max_points=1e6 , inter_weight=0.5, supervise_manner='3D'):
        super().__init__()
        self.ignore_classes = ignore_classes
        self.min_points = min_points
        self.max_points = max_points
        self.inter_weight = inter_weight
        self.supervise_manner = supervise_manner


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
    

        ndims = len(net_output.shape)
        if ndims == 5: # 3D
            if self.supervise_manner == '2D':
                # print("net_output", net_output.shape, 'target', target.shape)
                # net_output: B, C, D, H, W -> B, D, C, H, W
                net_output = net_output.permute(0, 2, 1, 3, 4).contiguous()
                net_output = net_output.reshape(
                    net_output.shape[0] * net_output.shape[1],
                    net_output.shape[2],
                    net_output.shape[3],
                    net_output.shape[4]
                )  # (B*D), C, H, W

                if target.ndim == 5:
                    # target: B, 1, D, H, W -> B, D, 1, H, W
                    target = target.permute(0, 2, 1, 3, 4).contiguous()
                    target = target.reshape(
                        target.shape[0] * target.shape[1],
                        target.shape[2],
                        target.shape[3],
                        target.shape[4]
                    )  # (B*D), 1, H, W

                    # if target.shape[1] == 1:
                    #     target = target.squeeze(1)  # (B*D), H, W

                elif target.ndim == 4:
                    # target: B, D, H, W
                    target = target.contiguous().reshape(
                        target.shape[0] * target.shape[1],
                        target.shape[2],
                        target.shape[3]
                    )  # (B*D), H, W


        # print("net_output", net_output.shape, 'target', target.shape)
        batch_size = net_output.shape[0]
        num_channels = net_output.shape[1]

        device = net_output.device

        class_means = []
        intra_losses = []
        
        for batch_i in range(batch_size):
            # each volume might have non sense ssame name id
                
            features = net_output[batch_i].view(1, num_channels, -1)
            features = features.permute(0, 2, 1).contiguous().view(-1, num_channels)  # (B*N), C
            labels = target[batch_i].view(1, -1).contiguous().view(-1)  # (B*N)

            unique_labels = torch.unique(labels)
            # print("unique_labels", unique_labels)

            for lab in unique_labels:
                if lab.item() in self.ignore_classes:
                    # print("unique lab", lab, 'ignore')
                    continue

                indices = (labels == lab).nonzero(as_tuple=True)[0]
                # print("min points label", lab.item(), indices.numel())

                if indices.numel() < self.min_points or indices.numel() > self.max_points:
                    
                    continue

                class_features = features[indices]                       # M, C
                class_mean = class_features.mean(dim=0, keepdim=True)    # 1, C

                # intra-class compactness
                sim_to_mean = F.cosine_similarity(class_features, class_mean, dim=1)
                intra_loss = 1.0 - sim_to_mean.mean()
                intra_losses.append(intra_loss)

                class_means.append(F.normalize(class_mean.squeeze(0), dim=0))


        if len(class_means) == 0:
            return torch.tensor(0.0, device=device)

        if len(intra_losses) == 0:
            return torch.tensor(0.0, device=device)

        intra_loss = torch.stack(intra_losses).mean()

        if len(class_means) < 2:
            return intra_loss

        class_means = torch.stack(class_means, dim=0)  # K, C
        sim_matrix = torch.matmul(class_means, class_means.T)

        eye = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        inter_loss = sim_matrix[~eye].mean()

        loss = intra_loss + self.inter_weight * inter_loss
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceSpatialContrastiveCompactLoss2D(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        compact_weight=1.0,
        normalize=True,
        ignore_ids=(0,),
        min_points=30,
        min_group_points=8,

    ):
        super().__init__()
        self.temperature = temperature
        self.compact_weight = compact_weight
        self.normalize = normalize
        self.ignore_ids = set(ignore_ids)
        self.min_points = min_points
        self.min_group_points = min_group_points

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dataset_ids=None):
        """
        net_output: B, C, H, W
        target:     B, 1, H, W
        target contains instance ids, 0 is usually background
        """
        # print("net_output", net_output.shape, 'target', target.shape)
        if len(net_output.shape) == 5:
            net_output = net_output.permute(0, 2, 1, 3, 4).contiguous()
            net_output = net_output.reshape(
                net_output.shape[0] * net_output.shape[1],
                net_output.shape[2],
                net_output.shape[3],
                net_output.shape[4]
            )  # (B*D), C, H, W

            target = target.permute(0, 2, 1, 3, 4).contiguous()
            target = target.reshape(
                target.shape[0] * target.shape[1],
                target.shape[2],
                target.shape[3],
                target.shape[4]
            )  # (B*D), 1, H, W


        B, C, H, W = net_output.shape
        device = net_output.device

        # coordinates shared across batch
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        coords = torch.stack([yy, xx], dim=-1).view(-1, 2).float()  # (H*W, 2)

        batch_losses = []

        for b in range(B):
            # -------- current batch only --------
            features_b = net_output[b].view(C, -1).T.contiguous()      # (H*W, C)
            instance_ids_b = target[b].view(-1)                        # (H*W,)

            unique_ids = torch.unique(instance_ids_b)

            aggregated_a = []
            aggregated_b = []
            compact_losses = []

            for inst_id in unique_ids:
                inst_int = int(inst_id.item())
                if inst_int in self.ignore_ids:
                    continue

                indices = (instance_ids_b == inst_id).nonzero(as_tuple=True)[0]
                if indices.numel() < self.min_points:
                    continue

                inst_features = features_b[indices]   # (M, C)
                inst_coords = coords[indices]         # (M, 2)

                # ---------- compactness ----------
                inst_mean = inst_features.mean(dim=0, keepdim=True)

                if self.normalize:
                    inst_features_norm = F.normalize(inst_features, dim=1)
                    inst_mean_norm = F.normalize(inst_mean, dim=1)
                    sim = F.cosine_similarity(inst_features_norm, inst_mean_norm, dim=1)
                    compact_loss = 1.0 - sim.mean()
                else:
                    compact_loss = F.l1_loss(inst_features, inst_mean.expand_as(inst_features))

                compact_losses.append(compact_loss)

                # ---------- spatial split ----------
                # print("inst_coords", inst_coords.shape)
                coord_var = inst_coords.var(dim=0)          # (2,)
                split_axis = torch.argmax(coord_var).item() # 0=h, 1=w

                split_values = inst_coords[:, split_axis]
                split_point = torch.median(split_values)

                mask_a = split_values <= split_point
                mask_b = split_values > split_point

                # fallback if median split degenerates
                if mask_a.sum() < self.min_group_points or mask_b.sum() < self.min_group_points:
                    sorted_idx = torch.argsort(split_values)
                    half = sorted_idx.numel() // 2

                    if half < self.min_group_points or (sorted_idx.numel() - half) < self.min_group_points:
                        continue

                    idx_a = sorted_idx[:half]
                    idx_b = sorted_idx[half:]
                    group_a = inst_features[idx_a]
                    group_b = inst_features[idx_b]
                else:
                    group_a = inst_features[mask_a]
                    group_b = inst_features[mask_b]

                if group_a.size(0) < self.min_group_points or group_b.size(0) < self.min_group_points:
                    continue

                agg_a = group_a.mean(dim=0)
                agg_b = group_b.mean(dim=0)

                aggregated_a.append(agg_a)
                aggregated_b.append(agg_b)

            # -------- no valid instance in this batch --------
            if len(compact_losses) == 0:
                continue

            compact_loss = torch.stack(compact_losses).mean()

            # if only one valid instance, only compactness is available
            if len(aggregated_a) < 2:
                loss_b = self.compact_weight * compact_loss
                batch_losses.append(loss_b)
                continue

            aggregated_a = torch.stack(aggregated_a, dim=0)  # (K, C)
            aggregated_b = torch.stack(aggregated_b, dim=0)  # (K, C)

            if self.normalize:
                aggregated_a = F.normalize(aggregated_a, dim=1)
                aggregated_b = F.normalize(aggregated_b, dim=1)

            logits_ab = torch.matmul(aggregated_a, aggregated_b.T) / self.temperature
            logits_ba = torch.matmul(aggregated_b, aggregated_a.T) / self.temperature

            targets = torch.arange(logits_ab.size(0), device=device)

            loss_a = F.cross_entropy(logits_ab, targets)
            loss_b2 = F.cross_entropy(logits_ba, targets)
            contrastive_loss = 0.5 * (loss_a + loss_b2)

            loss_b = contrastive_loss + self.compact_weight * compact_loss
            batch_losses.append(loss_b)

        if len(batch_losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(batch_losses).mean()




class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, **kwargs):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
