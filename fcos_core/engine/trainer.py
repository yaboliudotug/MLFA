# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# MLFA: Toward Realistic Test Time Adaptive Object Detection by Multi-Level Feature Alignment
# Modified by Yabo Liu
# Based on https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/engine/trainer.py
# and https://github.com/Gorilla-Lab-SCUT/TTAC/blob/master/cifar/TTAC_onepass.py
# --------------------------------------------------------

import os
import time
import numpy as np
import datetime
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist
from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.structures.image_list import to_image_list
from fcos_core.data.datasets.evaluation import validate
from .utils import covariance, PrototypeComputation, color_hub, compute_locations


def foward_detector(cfg, model, images, targets=None, return_maps=True,  DA_ON=True):

    with_rcnn = not cfg.MODEL.RPN_ONLY

    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()
    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    if model_fcos.training and DA_ON:
        losses = {}
        images_s, images_t = images
        features_s = model_backbone(images_s.tensors)
        features_t = model_backbone(images_t.tensors)
        proposals, proposal_losses, _ = model_fcos(
            images_s, features_s, targets=targets)

        # Convert feature representations
        f_s = {
            layer: features_s[map_layer_to_index[layer]]
            for layer in feature_layers
        }
        f_t = {
            layer: features_t[map_layer_to_index[layer]]
            for layer in feature_layers
        }

        losses.update(proposal_losses)

        return losses, (f_s, f_t)

    elif model_fcos.training  and not DA_ON:
        losses = {}
        features = model_backbone(images.tensors)
        if with_rcnn:
            proposals, proposal_losses = model_fcos(
                images, features, targets=targets)
            model_roi_head = model["roi_head"]
            feats, proposals, roi_head_loss = model_roi_head(features, proposals, targets)
            losses.update(proposal_losses)
            losses.update(roi_head_loss)
            return losses, []
        else:
            proposals, proposal_losses, score_maps = model_fcos(
                images, features, targets=targets)
            losses.update(proposal_losses)
            return losses, []

    else:

        images = to_image_list(images)
        features = model_backbone(images.tensors)
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

        return proposals
    

def forward_compute(cfg, model, images, targets=None, return_maps=True,  DA_ON=True, graph_generator=None):
    if targets is not None: # source domain
        model_backbone = model["backbone"]
        model_fcos = model["fcos"]
        model_backbone.eval()
        model_fcos.eval()
        features = model_backbone(images.tensors)
        features_selected, labels_selected, weights = graph_generator(compute_locations(features, cfg.MODEL.FCOS.FPN_STRIDES), features, targets)
        return features_selected, labels_selected, None, None
    else:   # target domain
        model_backbone = model["backbone"]
        model_fcos = model["fcos"]
        model_backbone.train()
        model_fcos.train()
        features = model_backbone(images.tensors.cuda())
        score_maps, zeros, feature_maps = model_fcos(images.tensors, features, targets=None, for_tta=True)
        score_maps_detach = [score_maps_detach.detach() for score_maps_detach in score_maps]
        features_selected, labels_selected, weights = graph_generator(None, features, score_maps_detach)
        return features_selected, labels_selected, score_maps, feature_maps


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def do_train(
        model,
        data_loader,
        optimizer,
        checkpointer,
        device,
        cfg,
):
    with_TTA = cfg.MODEL.TTA_ON
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    num_class_fgbg = cfg.MODEL.FCOS.NUM_CLASSES
    class_num = num_class_fgbg

    for k in model:
        model[k].train()
    start_training_time = time.time()
    logger.info('TTA_ON: {}'.format(str(with_TTA)))
    logger.info('device: {}'.format(device))

    data_loader_source = data_loader["source_4tta"]
    data_loader_target = data_loader["val_4tta"]

    # for ablation losses 
    # existing_dir = '/disk/liuyabo/research/ttaod_cluster/v_sigma_pp/experiments/ablation/losses/city_to_foggy_w' # city2foggy
    # existing_dir = '/disk/liuyabo/research/ttaod_cluster/v_sigma_pp/experiments/sigma_plus_plus_tta/kitti_to_cityscapes'  # kitti
    # existing_dir = '/disk/liuyabo/research/ttaod_cluster/v_sigma_pp/experiments/sigma_plus_plus_tta/sim10k_to_city'  # sim

    # existing_dir = '/disk/liuyabo/research/ttaod_cluster/tta_od_opensource/experiments/tta_od/city_to_foggy_ok'

    # compute source clusters
    existing_dir = cfg.OUTPUT_DIR
    offline_source_features_path = os.path.join(existing_dir, 'offline_source_features')
    offline_source_labels_path = os.path.join(existing_dir, 'offline_source_labels')
    graph_generator = PrototypeComputation(cfg)

    if os.path.exists(offline_source_features_path + '.npy') and os.path.exists(offline_source_labels_path + '.npy'):
        features_source = np.load(offline_source_features_path + '.npy')
        labels_source = np.load(offline_source_labels_path + '.npy')
        features_source = torch.from_numpy(features_source).to(device)
        labels_source = torch.from_numpy(labels_source).to(device)
    else:
        features_source = []
        labels_source = []
        logger.info('Start source clustering ...')
        with torch.no_grad():
            for idx, (images_s, targets_s, _) in enumerate(tqdm(data_loader_source)):
                images_s = images_s.to(device)
                targets_s = [target_s.to(device) for target_s in targets_s]
                try:
                    features_selected, labels_selected, _, _ = forward_compute(cfg, model, images_s, 
                                                                        targets=targets_s, graph_generator=graph_generator)
                except:
                    logger.info('index {} maybe empty'.format(idx))
                    continue
                features_source.append(features_selected)
                labels_source.append(labels_selected)
                # if idx > 50:
                #     break
        features_source = torch.cat(features_source, dim=0)
        labels_source = torch.cat(labels_source, dim=0)
        # np.save(offline_source_features_path, features_source.cpu().numpy())
        # np.save(offline_source_labels_path, labels_source.cpu().numpy())
        logger.info('Finish source clustering ...')
    
    ext_src_mu = []
    ext_src_cov = []
    for k in range(num_class_fgbg):
        index_k = labels_source == k
        features_source_k = features_source[index_k]
        ext_src_mu.append(features_source_k.mean(0))
        ext_src_cov.append(covariance(features_source_k))

    mu_src_ext = features_source.mean(dim=0)
    cov_src_ext = covariance(features_source)

    bias = cov_src_ext.max().item() / 30.
    template_ext_cov = torch.eye(256).cuda() * bias

    ext_src_mu = torch.stack(ext_src_mu)
    ext_src_cov = torch.stack(ext_src_cov)

    ema_ext_mu = ext_src_mu.clone()
    ema_ext_cov = ext_src_cov.clone()
    ema_ext_total_mu = torch.zeros(256).float()
    ema_ext_total_cov = torch.zeros(256, 256).float()

    ema_n = torch.zeros(class_num).cuda()
    ema_total_n = 0.
    ema_length = 128

    results_dict = {}
    dataset_target = data_loader_target.dataset
    count_idx = 0
    for mini_batch_idx, (images_t, targets_t, indexes_t) in enumerate(data_loader_target):
        count_idx += 1
        for k in optimizer:
            optimizer[k].zero_grad()
        loss_dict = {}
        images_t.to(device)
        targets_t = [targets_t.to(device) for targets_t in targets_t]
        features_selected, labels_selected, score_maps, feature_maps = forward_compute(cfg, model, images_t, 
                                                                targets=None, graph_generator=graph_generator)
        box_cls, box_regression, centerness = feature_maps

        if features_selected is None:
            results_dict.update(
                {img_id: [] for img_id in indexes_t}
            )
            continue

        if cfg.MODEL.TTA_loss_kl_category:
            loss_kl_category = 0
            feat_ext2 = features_selected
            pseudo_label2 = labels_selected
            for label in pseudo_label2.unique():                        
                feat_ext_per_category = feat_ext2[pseudo_label2 == label, :]

                if feat_ext_per_category.shape[0] > 100:
                    max_length = 100
                    feat_ext_per_category = feat_ext_per_category[:max_length]

                b = feat_ext_per_category.shape[0]
                ema_n[label] += b
                alpha = 1. / ema_length if ema_n[label] > ema_length else 1. / ema_n[label]

                ema_ext_mu_that = ema_ext_mu[label, :]
                ema_ext_cov_that = ema_ext_cov[label, :, :]
                delta_pre = feat_ext_per_category - ema_ext_mu_that

                delta = alpha * delta_pre.sum(dim=0)
                tmp_mu = ema_ext_mu_that + delta
                tmp_cov = ema_ext_cov_that + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_cov_that) - delta[:, None] @ delta[None, :]

                with torch.no_grad():
                    ema_ext_mu[label, :] = tmp_mu.detach()
                    ema_ext_cov[label, :, :] = tmp_cov.detach()
                
                if ema_n[label] >= 16:
                    try:
                        source_domain = torch.distributions.MultivariateNormal(ext_src_mu[label, :], ext_src_cov[label, :, :] + template_ext_cov)
                        target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
                        loss_kl_category += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) / 2.0
                    except:
                        logger.info('wrong kl category label: ' + str(label))
                        continue
            loss_dict['loss_kl_category'] = loss_kl_category / len(pseudo_label2.unique()) * cfg.MODEL.loss_kl_category_scale


        if cfg.MODEL.TTA_loss_kl_global:
            feat_ext = features_selected
            b = feat_ext.shape[0]
            ema_total_n += b
            alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
            delta_pre = (feat_ext - ema_ext_total_mu.cuda())
            delta = alpha * delta_pre.sum(dim=0)
            tmp_mu = ema_ext_total_mu.cuda() + delta
            tmp_cov = ema_ext_total_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_total_cov.cuda()) - delta[:, None] @ delta[None, :]
            with torch.no_grad():
                ema_ext_total_mu = tmp_mu.detach().cpu()
                ema_ext_total_cov = tmp_cov.detach().cpu()

            source_domain = torch.distributions.MultivariateNormal(mu_src_ext, cov_src_ext + template_ext_cov)
            target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
            loss_kl_global = (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) / 2.
            loss_dict['loss_kl_global'] = loss_kl_global * cfg.MODEL.loss_kl_global_scale


        if cfg.MODEL.TTA_loss_entropy_category:
            box_cls_flatten = []
            for l in range(len(box_cls)):
                box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_class_fgbg-1))
            box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
            loss_entropy_category = softmax_entropy(box_cls_flatten).mean()
            loss_dict['entropy_category_loss'] = loss_entropy_category * cfg.MODEL.loss_entropy_category_scale
        

        losses = sum(loss for loss in loss_dict.values())
        logger.info('iter: {}, loss: {}'.format(mini_batch_idx, losses))
        if losses > 0:
            losses.backward()
            del loss_dict, losses
            for k in optimizer:
                optimizer[k].step()

        with torch.no_grad():
            model_backbone = model["backbone"].eval()
            model_fcos = model["fcos"].eval()
            images_t = to_image_list(images_t)
            features = model_backbone(images_t.tensors.cuda())
            output, _, _ = model_fcos(images_t, features, targets=None, return_maps=False)
        cpu_device = torch.device("cpu")
        output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(indexes_t, output)}
        )


    extra_args = dict(
        box_only=False,
        iou_types=("bbox",),
        expected_results=(),
        expected_results_sigma_tol=4,
    )

    image_ids = list(sorted(results_dict.keys()))
    predictions = [results_dict[i] for i in image_ids]

    val_results = validate(dataset=dataset_target,
            predictions=predictions,
            output_folder=None,
            **extra_args)

    meter_AP50= val_results[0].results['bbox']['AP50'] * 100
    meter_AP = val_results[0].results['bbox']['AP']* 100
    logger.info('Final bbox results (AP): {}'.format(str(meter_AP)))
    logger.info('Final bbox results (AP50): {}'.format(str(meter_AP50)))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total test time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (len(data_loader_target))))
    # checkpointer.save("model_final")
